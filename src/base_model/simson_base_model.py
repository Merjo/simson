import numpy as np
import pandas as pd
import os
import sys
import pickle
from ODYM.odym.modules.ODYM_Classes import MFAsystem, Classification, Process, Parameter
from src.odym_extension.SimDiGraph_MFAsystem import SimDiGraph_MFAsystem
from src.tools.config import cfg
from src.base_model.model_tools import get_dsm_data, get_stock_data_country_specific_areas, calc_change_timeline
from src.base_model.load_dsms import load_dsms
from src.base_model.load_params import get_cullen_fabrication_yield, get_cullen_forming_yield, \
    get_wittig_distributions, get_worldsteel_recovery_rates, get_daehn_external_copper_rate, \
    get_daehn_tolerances, get_daehn_good_intermediate_distribution, get_cullen_production_yield
from src.calc_trade.calc_trade import get_trade
from src.calc_trade.calc_scrap_trade import get_scrap_trade
from src.calc_trade.calc_indirect_trade import get_indirect_trade
from src.calc_trade.calc_trade_tools import get_imports_and_exports_from_net_trade
from src.recycling_strategies.base_model import lower_cycle_base_model
from src.recycling_strategies.tramp_elements import lower_cycle_tramp_elements
from src.recycling_strategies.economic_tramp_elements import lower_cycle_econ_tramp
from src.modelling_approaches.compute_upper_cycle import compute_upper_cycle
from src.modelling_approaches.model_1_inflow_driven import distribute_intermediate_good, calc_lifetime_matrix
from src.read_data.load_data import load_lifetimes
from src.base_model.tramp_econ_model import calc_tramp_econ_model
from src.economic_model.econ_model_tools import get_steel_prices

#  constants: MFA System process IDs

ENV_PID = 0
BOF_PID = 1
EAF_PID = 2
FORM_PID = 3
IP_PID = 4
FABR_PID = 5
USE_PID = 6
BUF_PID = 7
OBS_PID = 8
EOL_PID = 9
RECYCLE_PID = 10
SCRAP_PID = 11
EXC_PID = 12
FBUF_PID = 13


def load_simson_base_model(country_specific=False, recalculate=False, recalculate_dsms=False) -> SimDiGraph_MFAsystem:
    file_name_end = 'countries' if country_specific else f'{cfg.region_data_source}_regions'
    file_name = f'main_model_{file_name_end}.p'
    file_path = os.path.join(cfg.data_path, 'models', file_name)
    do_load_existing = os.path.exists(file_path) and not recalculate
    if do_load_existing:
        model = pickle.load(open(file_path, "rb"))
    else:
        model = create_base_model(country_specific, recalculate_dsms)
        pickle.dump(model, open(file_path, "wb"))
    return model


def create_base_model(country_specific, recalculate_dsms):
    # dsms = load_dsms(country_specific, recalculate_dsms)
    model, balance_message = create_model(country_specific)
    print(balance_message)
    return model


def create_model(country_specific, scrap_share_in_production=None):
    n_regions = 12
    max_scrap_share_in_production = _calc_max_scrap_share(scrap_share_in_production, n_regions)
    # load data
    areas = get_stock_data_country_specific_areas(country_specific)
    main_model = set_up_model(areas)
    # Load base_model
    initiate_model(main_model)

    # compute stocks and flows
    inflows, outflows = compute_flows(main_model, country_specific,
                                      max_scrap_share_in_production)
    compute_stocks(main_model, inflows, outflows)

    # check base_model
    balance_message = mass_balance_plausible(main_model)

    return main_model, balance_message


def initiate_model(main_model):
    initiate_processes(main_model)
    initiate_parameters(main_model)
    initiate_flows(main_model)
    initiate_stocks(main_model)
    main_model.Initialize_FlowValues()
    main_model.Initialize_StockValues()
    check_consistency(main_model)


def set_up_model(regions):
    model_classification = {'Time': Classification(Name='Time', Dimension='Time', ID=1,
                                                   Items=cfg.years),
                            'Element': Classification(Name='Elements', Dimension='Element', ID=2, Items=cfg.elements),
                            'Region': Classification(Name='Regions', Dimension='Region', ID=3, Items=regions),
                            'Intermediate': Classification(Name='Intermediate', Dimension='Material',
                                                           ID=4, Items=cfg.intermediate_products),
                            'Good': Classification(Name='Goods', Dimension='Material', ID=5,
                                                   Items=cfg.in_use_categories),
                            'Scenario': Classification(Name='Scenario', Dimension='Scenario', ID=6,
                                                       Items=cfg.scenarios),
                            'Production': Classification(Name='Production', Dimension='Production', ID=7,
                                                         Items=['BOF', 'EAF'])}
    model_time_start = cfg.start_year
    model_time_end = cfg.end_year
    index_table = pd.DataFrame(
        {'Aspect': ['Time', 'Element', 'Region', 'Intermediate', 'Good', 'Scenario', 'Production'],
         'Description': ['Model aspect "Time"', 'Model aspect "Element"',
                         'Model aspect "Region"', 'Model aspect "Intermediate"',
                         'Model aspect "Good"', 'Model aspect "Scenario"',
                         'Model aspect "Production"'],
         'Dimension': ['Time', 'Element', 'Region', 'Material', 'Material',
                       'Scenario', 'Production'],
         'Classification': [model_classification[Aspect] for Aspect in
                            ['Time', 'Element', 'Region', 'Intermediate', 'Good', 'Scenario',
                             'Production']],
         'IndexLetter': ['t', 'e', 'r', 'i', 'g', 's', 'p']})
    index_table.set_index('Aspect', inplace=True)

    main_model = SimDiGraph_MFAsystem(Name='SIMSON',
                                      Geogr_Scope='World',
                                      Unit='t',
                                      ProcessList=[],
                                      FlowDict={},
                                      StockDict={},
                                      ParameterDict={},
                                      Time_Start=model_time_start,
                                      Time_End=model_time_end,
                                      IndexTable=index_table,
                                      Elements=index_table.loc['Element'].Classification.Items)

    return main_model


def initiate_processes(main_model):
    main_model.ProcessList = []

    def add_process(name, p_id):
        main_model.ProcessList.append(Process(Name=name, ID=p_id))

    add_process('Primary Production / Environment', ENV_PID)
    add_process('BF/BOF Production', BOF_PID)
    add_process('EAF Production', EAF_PID)
    add_process('Forming', FORM_PID)
    add_process('Intermediate', IP_PID)
    add_process('Fabrication', FABR_PID)
    add_process('Using', USE_PID)
    add_process('Outflow Buffer', BUF_PID)
    add_process('Obsolete Stock', OBS_PID)
    add_process('End of Life', EOL_PID)
    add_process('Recycling', RECYCLE_PID)
    add_process('Scrap Market', SCRAP_PID)
    add_process('Excess Scrap', EXC_PID)
    add_process('Fabrication Buffer', FBUF_PID)


def initiate_parameters(main_model):
    parameter_dict = {}

    fabrication_yield = get_cullen_fabrication_yield()
    forming_yield = get_cullen_forming_yield()
    recovery_rate = get_worldsteel_recovery_rates()
    copper_rate = get_daehn_external_copper_rate()
    tolerances = get_daehn_tolerances()
    gi_distribution = get_daehn_good_intermediate_distribution()
    production_yield = get_cullen_production_yield()

    parameter_dict['Forming_Yield'] = Parameter(Name='Forming_Yield', ID=0,
                                                P_Res=FORM_PID, MetaData=None, Indices='i',
                                                Values=forming_yield, Unit='1')

    parameter_dict['Fabrication_Yield'] = Parameter(Name='Fabrication_Yield', ID=1,
                                                    P_Res=FABR_PID, MetaData=None, Indices='g',
                                                    Values=fabrication_yield, Unit='1')

    parameter_dict['Production_Yield'] = Parameter(Name='Production_Yield', ID=2,
                                                   P_Res=FABR_PID, MetaData=None, Indices='p',
                                                   Values=production_yield, Unit='1')

    parameter_dict['Recovery_Rate'] = Parameter(Name='Recovery Rate', ID=3,
                                                P_Res=EOL_PID,
                                                MetaData=None, Indices='g',
                                                Values=recovery_rate, Unit='1')

    parameter_dict['External_Copper_Rate'] = Parameter(Name='External_Copper_Rate', ID=4,
                                                       P_Res=RECYCLE_PID,
                                                       MetaData=None, Indices='g',
                                                       Values=copper_rate, Unit='1')

    parameter_dict['Copper_Tolerances'] = Parameter(Name='Copper_Tolerances', ID=5,
                                                    P_Res=IP_PID,
                                                    MetaData=None, Indices='i',
                                                    Values=tolerances, Unit='1')

    parameter_dict['Good_Intermediate_Distribution'] = Parameter(Name='Good_Intermediate_Distribution', ID=6,
                                                                 P_Res=IP_PID,
                                                                 MetaData=None, Indices='gi',
                                                                 Values=gi_distribution, Unit='1')

    main_model.ParameterDict = parameter_dict


def initiate_flows(main_model):
    main_model.init_flow('Iron production - BOF', ENV_PID, BOF_PID, 't,e,r,s')
    main_model.init_flow('Scrap - BOF', SCRAP_PID, BOF_PID, 't,e,r,s')
    main_model.init_flow('BOF - Forming', BOF_PID, FORM_PID, 't,e,r,s')
    main_model.init_flow('Scrap - EAF', SCRAP_PID, EAF_PID, 't,e,r,s')
    main_model.init_flow('EAF scaler - Forming', EAF_PID, FORM_PID, 't,e,r,s')

    main_model.init_flow('Forming - IP Market', FORM_PID, IP_PID, 't,e,r,i,s')
    main_model.init_flow('Forming - Fabrication Buffer', FORM_PID, FBUF_PID, 't,e,r,s')
    main_model.init_flow('IP Market - Fabrication', IP_PID, FABR_PID, 't,e,r,i,s')
    main_model.init_flow('Fabrication - In-Use', FABR_PID, USE_PID, 't,e,r,g,s')
    main_model.init_flow('Fabrication - Fabrication Buffer', FABR_PID, FBUF_PID, 't,e,r,s')
    main_model.init_flow('Fabrication Buffer - Scrap', FBUF_PID, SCRAP_PID, 't,e,r,s')

    main_model.init_flow('In-Use - Outflow Buffer', USE_PID, BUF_PID, 't,e,r,g,s')
    main_model.init_flow('Outflow Buffer - Obsolete stocks', BUF_PID, OBS_PID, 't,e,r,g,s')
    main_model.init_flow('Outflow Buffer - EOL', BUF_PID, EOL_PID, 't,e,r,g,s')
    main_model.init_flow('EOL - Recycling', EOL_PID, RECYCLE_PID, 't,e,r,g,s')
    main_model.init_flow('Copper - Recyling', ENV_PID, RECYCLE_PID, 't,e,r,g,s')
    main_model.init_flow('Recycling - Scrap', RECYCLE_PID, SCRAP_PID, 't,e,r,g,s')
    main_model.init_flow('Scrap - Waste', SCRAP_PID, EXC_PID, 't,e,r,s')

    # Trade

    main_model.init_flow('Crude Imports', ENV_PID, IP_PID, 't,e,r,i,s')
    main_model.init_flow('Crude Exports', IP_PID, ENV_PID, 't,e,r,i,s')
    main_model.init_flow('EoL Imports', ENV_PID, EOL_PID, 't,e,r,g,s')
    main_model.init_flow('EoL Exports', EOL_PID, ENV_PID, 't,e,r,g,s')
    main_model.init_flow('Indirect Imports', ENV_PID, USE_PID, 't,e,r,g,s')
    main_model.init_flow('Indirect Exports', USE_PID, ENV_PID, 't,e,r,g,s')


def initiate_stocks(main_model):
    main_model.add_stock(USE_PID, 'In-Use stock', 't,e,r,g,s')
    main_model.add_stock(BUF_PID, 'Outflow Buffer', 't,e,r,g,s')
    main_model.add_stock(FBUF_PID, 'Forming/Fabrication Buffer', 't,e,r,s')
    main_model.add_stock(OBS_PID, 'Obsolete stock', 't,e,r,g,s')
    main_model.add_stock(EXC_PID, 'Excess scrap stock', 't,e,r,s')


def check_consistency(main_model: MFAsystem):
    """
    Uses ODYM consistency checks to see if base_model dimensions and structure are well
    defined. Raises RuntimeError if not.

    :param main_model: The MFA System
    :return:
    """
    consistency = main_model.Consistency_Check()
    for consistencyCheck in consistency:
        if not consistencyCheck:
            raise RuntimeError("A consistency check failed: " + str(consistency))


def compute_flows(model: MFAsystem, country_specific: bool, max_scrap_share_in_production: np.ndarray):
    """

    :param model: The MFA system
    :param country_specific:
    :param inflows:
    :param outflows:
    :param max_scrap_share_in_production:
    :return:
    """

    # Compute upper cycle
    # production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows

    forming_yield, fabrication_yield, production_yield, recovery_rate, ext_copper_rate, tolerances, gi_distribution = \
        _get_params(model)

    production_yield = np.average(production_yield)

    reuse = None  # TODO: necessary?
    # TODO: Decide Reuse what to do
    """
    if cfg.do_change_reuse and not cfg.do_model_approaches:  # TODO: model approaches with reuse?
        # one is substracted as one was added to multiply scenario and category reuse changes
        reuse_factor_timeline = calc_change_timeline(cfg.reuse_factor, cfg.reuse_change_base_year) - 1
        reuse = np.einsum('trgs,tgs->trgs', outflows, reuse_factor_timeline)
        inflows -= reuse
        outflows -= reuse
    """

    production, forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use, \
    indirect_imports, indirect_exports, inflows, stocks, outflows = compute_upper_cycle_modelling_approaches()
    forming_scrap = np.einsum('tris,i->trs', forming_intermediate, (1 / forming_yield - 1))

    fabrication_scrap = np.einsum('trgs,g->trs', fabrication_use, (1 / fabrication_yield - 1))

    fabrication_buffer = np.zeros_like(fabrication_scrap)
    fabrication_buffer[1:] = forming_scrap[:-1] + fabrication_scrap[:-1]
    outflow_buffer = np.zeros_like(outflows)
    outflow_buffer[1:] = outflows[:-1]
    # fabrication_buffer = forming_scrap + fabrication_scrap
    # outflow_buffer = outflows
    """
    iron_production, scrap_in_bof, bof_production, eaf_production, use_eol, use_obsolete, scrap_imports, \
    scrap_exports, eol_recycling, recycling_scrap, scrap_excess = \
        _compute_lower_cycle(country_specific, outflow_buffer, recovery_rate, copper_rate, production, forming_yield,
                             forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use,
                             indirect_imports, indirect_exports,
                             max_scrap_share_in_production, forming_scrap, fabrication_scrap)"""

    production_by_intermediate = np.einsum('tris,i->tris', forming_intermediate, 1 / forming_yield)

    cu_recycling_scrap, cu_scrap_in_production, cu_production, cu_production_by_intermediate, \
    cu_forming_intermediate, cu_forming_scrap, cu_imports, cu_exports, cu_intermediate_fabrication, \
    cu_fabrication_scrap, cu_fabrication_use, cu_indirect_imports, cu_indirect_exports, \
    cu_inflows, cu_stocks, cu_outflows, cu_buffer, cu_fabrication_buffer, \
    cu_buffer_eol, cu_scrap_exports, cu_scrap_imports, cu_eol_recycling, \
    cu_available_scrap, cu_tolerated, \
    cu_iron_production, cu_bof_production, cu_eaf_production, cu_scrap_in_bof = \
        _create_cu_flows(production, fabrication_use, forming_intermediate)

    do_econ_model = cfg.do_model_economy
    if do_econ_model:
        q_st = production_by_intermediate / production_yield
        q_eol = outflow_buffer
        buffer_eol_best_guess = np.einsum(f'trgs,g->trgs', outflow_buffer, recovery_rate)
        scrap_imports, scrap_exports = get_scrap_trade(country_specific=country_specific,
                                                       available_scrap_by_category=buffer_eol_best_guess)
        # TODO: note - in  econ model, scrpa trade is scaled by BUFFER not available scrap after recycling as this
        #  shall be calculated via the econ moddle

        ip_tlrc_i = tolerances
        t_eol_g = scrap_imports - scrap_exports
        q_eol_total = np.sum(q_eol, axis=2)
        t_eol_share = np.divide(np.sum(t_eol_g, axis=2),
                                q_eol_total,
                                out=np.zeros_like(production),
                                where=q_eol_total != 0)
        # calculation of s_cu_max
        s_cu_max_numerator = np.einsum('tris,i->tris', production_by_intermediate, ip_tlrc_i)
        sum_numerator = np.einsum('tris->trs', s_cu_max_numerator)
        sum_denominator = np.einsum('tris->trs', production_by_intermediate)
        s_cu_max = sum_numerator / sum_denominator

        q_st_total = np.sum(q_st, axis=2)

        q_primary_scrap = fabrication_buffer

        q_pr_st, q_se_st, econ_recovery_rate, total_copper_rate = calc_tramp_econ_model(q_st_total, q_eol,
                                                                                        q_primary_scrap,
                                                                                        t_eol_share, s_cu_max)

    econ_start_index = cfg.econ_start_index
    buffer_eol = np.einsum(f'trgs,g->trgs', outflow_buffer, recovery_rate)
    if do_econ_model:
        buffer_eol[econ_start_index:] = np.einsum(f'trgs,trgs->trgs',
                                                  outflow_buffer[econ_start_index:],
                                                  econ_recovery_rate[econ_start_index:])
    buffer_obsolete = outflow_buffer - buffer_eol

    scrap_imports, scrap_exports = get_scrap_trade(country_specific=country_specific,
                                                   available_scrap_by_category=buffer_eol)

    total_eol_scrap = buffer_eol + scrap_imports - scrap_exports
    scrap_is_positive = np.all(total_eol_scrap[124:] >= 0)
    if not scrap_is_positive:
        raise RuntimeError('Scrap is not positive')
    eol_recycling = total_eol_scrap
    recycling_scrap = total_eol_scrap
    cu_external = np.einsum(f'trgs,g->trgs', eol_recycling, ext_copper_rate)  # is changed for econ model
    available_scrap = np.sum(recycling_scrap, axis=2) + fabrication_buffer
    scrap_in_production = np.zeros_like(production)
    scrap_used_rate = np.zeros_like(production)

    intermediate_fabrication_by_ig = np.einsum('trgs,g,gi->trigs', fabrication_use,
                                               1 / fabrication_yield, gi_distribution)
    intermediate_fabrication_sum = np.sum(intermediate_fabrication_by_ig, axis=3)
    intermediate_fabrication_divisor = np.divide(1, intermediate_fabrication_sum,
                                                 out=np.zeros_like(intermediate_fabrication_sum),
                                                 where=intermediate_fabrication_sum != 0)
    intermediate_fabrication_ig_share = np.einsum('trigs,tris->trigs',
                                                  intermediate_fabrication_by_ig,
                                                  intermediate_fabrication_divisor)
    mean, std_dev = load_lifetimes()
    lifetime_matrix = calc_lifetime_matrix(mean, std_dev, n_years=201)

    for t in range(1, 201):
        do_econ_model_this_year = False
        if t >= econ_start_index:
            do_econ_model_this_year = True

        cu_buffer[t] = cu_outflows[t - 1]
        cu_fabrication_buffer[t] = cu_forming_scrap[t - 1] + cu_fabrication_scrap[t - 1]
        current_recovery_rate = econ_recovery_rate[t] if do_econ_model_this_year else recovery_rate
        recovery_rate_dims = 'rgs' if do_econ_model_this_year else 'g'
        cu_buffer_eol[t] = np.einsum(f'rgs,{recovery_rate_dims}->rgs', cu_buffer[t], current_recovery_rate)

        if do_econ_model_this_year:
            a = 0

        cu_scrap_imports[t], cu_scrap_exports[t] = _calc_cu_trade(cu_buffer_eol[t], buffer_eol[t],
                                                                  scrap_imports[t], scrap_exports[t])

        if do_econ_model_this_year:
            a = 0
        cu_total_eol_scrap_t = cu_buffer_eol[t] + cu_scrap_imports[t] - cu_scrap_exports[t]
        cu_eol_recycling[t] = cu_total_eol_scrap_t
        if do_econ_model_this_year:
            cu_external[t] = _calc_econ_cu_external(total_copper_rate[t], ext_copper_rate, total_eol_scrap[t],
                                                    cu_total_eol_scrap_t)
        cu_recycling_scrap[t] = cu_total_eol_scrap_t + cu_external[t]
        cu_available_scrap[t] = cu_fabrication_buffer[t] + np.sum(cu_recycling_scrap[t], axis=1)
        cu_tolerated[t] = np.einsum('ris,i->ris', production_by_intermediate[t], tolerances)
        cu_tolerated_sum_t = np.sum(cu_tolerated[t], axis=1)
        if do_econ_model_this_year:
            scrap_in_production[t] = np.minimum(available_scrap[t], production[t])
            scrap_used_rate[t] = np.divide(scrap_in_production[t], available_scrap[t],
                                           out=np.zeros_like(scrap_in_production[t]),
                                           where=available_scrap[t] != 0)
            cu_scrap_in_production[t] = cu_available_scrap[t] * scrap_used_rate[t]
        elif cfg.recycling_strategy == 'tramp':
            cu_scrap_in_production[t] = np.minimum(cu_available_scrap[t], cu_tolerated_sum_t)
            scrap_used_rate[t] = np.divide(cu_scrap_in_production[t], cu_available_scrap[t],
                                           out=np.zeros_like(cu_scrap_in_production[t]),
                                           where=cu_available_scrap[t] != 0)
            scrap_in_production[t] = available_scrap[t] * scrap_used_rate[t]
        elif cfg.recycling_strategy == 'base':
            max_scrap_in_production_t = production[t] * max_scrap_share_in_production[t]
            scrap_in_production[t] = np.minimum(available_scrap[t], max_scrap_in_production_t)
            scrap_used_rate[t] = np.divide(scrap_in_production[t], available_scrap[t],
                                           out=np.zeros_like(scrap_in_production[t]),
                                           where=available_scrap[t] != 0)
            cu_scrap_in_production[t] = cu_available_scrap[t] * scrap_used_rate[t]
        else:
            raise RuntimeError(f'Recycling strategy has to be base or tramp or econ, not {cfg.recycling_strategy}')

        cu_tolerated_share_t = np.divide(cu_scrap_in_production[t], cu_tolerated_sum_t,
                                         out=np.zeros_like(cu_scrap_in_production[t]),
                                         where=cu_tolerated_sum_t != 0)
        cu_scrap_production_intermediate_t = np.einsum('ris,rs->ris', cu_tolerated[t], cu_tolerated_share_t)

        # test cu_scrap sector split
        test_cu_sector_split = np.sum(cu_scrap_production_intermediate_t, axis=1) - cu_scrap_in_production[t]
        if not np.all(test_cu_sector_split < 1):
            raise RuntimeError('Copper sector split not equal to total copper scrap in production.')

        cu_forming_intermediate[t] = np.einsum('ris,i->ris', cu_scrap_production_intermediate_t, forming_yield)
        cu_forming_scrap[t] = np.sum(cu_scrap_production_intermediate_t - cu_forming_intermediate[t], axis=1)

        cu_imports[t], cu_exports[t] = _calc_cu_trade(cu_forming_intermediate[t], forming_intermediate[t],
                                                      imports[t], exports[t])

        cu_intermediate_fabrication[t] = cu_forming_intermediate[t] + cu_imports[t] - cu_exports[t]

        cu_fabrication_outflow_t = np.einsum('ris,rigs->rgs',
                                             cu_intermediate_fabrication[t],
                                             intermediate_fabrication_ig_share[t])

        # test fabrication balance
        test_fabrication_balance = np.sum(cu_intermediate_fabrication[t], axis=1) - np.sum(cu_fabrication_outflow_t,
                                                                                           axis=1)
        test_fabrication_balance2 = np.all(np.abs(test_fabrication_balance) < 1)
        if not test_fabrication_balance2:
            raise RuntimeError('Copper fabrication balance wrong.')

        cu_fabrication_use[t] = np.einsum('rgs,g->rgs', cu_fabrication_outflow_t, fabrication_yield)
        cu_fabrication_scrap[t] = np.sum(cu_fabrication_outflow_t - cu_fabrication_use[t], axis=1)
        cu_indirect_imports[t], cu_indirect_exports[t] = _calc_cu_trade(cu_fabrication_use[t], fabrication_use[t],
                                                                        indirect_imports[t], indirect_exports[t])
        cu_inflows[t] = cu_fabrication_use[t] + cu_indirect_imports[t] - cu_indirect_exports[t]
        cu_outflows[t] = np.einsum('trgs,trg->rgs', cu_inflows[:t + 1], lifetime_matrix[t, :t + 1])
        cu_stocks[t] = cu_stocks[t - 1] + cu_inflows[t] - cu_outflows[t]

    scrap_share = np.divide(scrap_in_production, production,
                            out=np.zeros_like(scrap_in_production), where=production != 0)
    eaf_share_production = _calc_eaf_share_production(scrap_share)
    eaf_production = production * eaf_share_production
    bof_production = production - eaf_production
    max_scrap_in_bof = cfg.scrap_in_BOF_rate * bof_production
    scrap_in_bof = np.minimum(max_scrap_in_bof, scrap_in_production)
    inv_scrap_in_production = np.divide(1, scrap_in_production,
                                        out=np.zeros_like(scrap_in_production),
                                        where=scrap_in_production != 0)
    scrap_in_bof_rate = np.einsum('trs,trs->trs', scrap_in_bof, inv_scrap_in_production)
    if np.any(scrap_in_bof_rate < -0) or np.any(scrap_in_bof_rate > 1):
        raise RuntimeError('Error in scrap in bof rate calculation.')
    cu_scrap_in_bof = cu_scrap_in_production * scrap_in_bof_rate
    cu_eaf_production = cu_scrap_in_production - cu_scrap_in_bof
    cu_bof_production = cu_scrap_in_bof
    iron_production = bof_production - scrap_in_bof

    scrap_in_production = scrap_in_bof + eaf_production
    scrap_excess = available_scrap - scrap_in_production
    cu_scrap_excess = cu_available_scrap - cu_scrap_in_production
    cu_buffer_obsolete = cu_buffer - cu_buffer_eol

    # join together
    iron_production = np.stack([iron_production, cu_iron_production], axis=1)
    scrap_in_bof = np.stack([scrap_in_bof, cu_scrap_in_bof], axis=1)
    bof_production = np.stack([bof_production, cu_bof_production], axis=1)
    eaf_production = np.stack([eaf_production, cu_eaf_production], axis=1)
    forming_intermediate = np.stack([forming_intermediate, cu_forming_intermediate], axis=1)
    forming_scrap = np.stack([forming_scrap, cu_forming_scrap], axis=1)
    intermediate_fabrication = np.stack([intermediate_fabrication, cu_intermediate_fabrication], axis=1)
    imports = np.stack([imports, cu_imports], axis=1)
    exports = np.stack([exports, cu_exports], axis=1)
    fabrication_use = np.stack([fabrication_use, cu_fabrication_use], axis=1)
    indirect_imports = np.stack([indirect_imports, cu_indirect_imports], axis=1)
    indirect_exports = np.stack([indirect_exports, cu_indirect_exports], axis=1)
    # reuse not included thus far
    fabrication_scrap = np.stack([fabrication_scrap, cu_fabrication_scrap], axis=1)
    outflows = np.stack([outflows, cu_outflows], axis=1)
    buffer_eol = np.stack([buffer_eol, cu_buffer_eol], axis=1)
    buffer_obsolete = np.stack([buffer_obsolete, cu_buffer_obsolete], axis=1)
    scrap_imports = np.stack([scrap_imports, cu_scrap_imports], axis=1)
    scrap_exports = np.stack([scrap_exports, cu_scrap_exports], axis=1)
    eol_recycling = np.stack([eol_recycling, cu_eol_recycling], axis=1)
    recycling_scrap = np.stack([recycling_scrap, cu_recycling_scrap], axis=1)
    scrap_excess = np.stack([scrap_excess, cu_scrap_excess], axis=1)
    fabrication_buffer = np.stack([fabrication_buffer, cu_fabrication_buffer], axis=1)
    env_recycling = np.stack([np.zeros_like(cu_external), cu_external], axis=1)

    edit_flows(model, iron_production, scrap_in_bof, bof_production, eaf_production, forming_intermediate,
               forming_scrap, intermediate_fabrication, imports, exports, fabrication_use, indirect_imports,
               indirect_exports, reuse, fabrication_scrap, outflows, buffer_eol, buffer_obsolete, scrap_imports,
               scrap_exports, eol_recycling, recycling_scrap, scrap_excess, fabrication_buffer, env_recycling)

    return inflows, outflows


def _calc_econ_cu_external(total_copper_rate, ext_copper_rate, scrap, cu_internal_scrap):
    total_copper = np.einsum('rs,rgs->rs', total_copper_rate, scrap)
    factor = np.einsum('g,h->gh', ext_copper_rate, 1 / ext_copper_rate)  # 'h' denotes another axis of the 'g' dimension
    divisor = np.einsum('rgs,hg->rhs', scrap, factor)
    inv_divisor = np.divide(1, divisor, out=np.zeros_like(divisor), where=divisor != 0)
    copper_rate_g = np.einsum('rs,rgs->rgs', total_copper, inv_divisor)
    total_copper_g = np.einsum('rgs,rgs->rgs', copper_rate_g, scrap)
    cu_external = total_copper_g - cu_internal_scrap

    return cu_external


def _calc_cu_trade(cu_inflow, inflow, imports, exports):
    export_share = np.divide(exports, inflow, out=np.zeros_like(exports), where=inflow != 0)
    cu_exports = export_share * cu_inflow
    cu_global_exports = np.sum(cu_exports, axis=0)

    global_imports = np.sum(imports, axis=0)
    inv_global_imports = np.divide(1, global_imports, out=np.zeros_like(global_imports), where=global_imports != 0)
    import_share = np.einsum('rgs,gs->rgs', imports, inv_global_imports)  # g can be i for crude imports/exports
    cu_imports = np.einsum('rgs,gs->rgs', import_share, cu_global_exports)  # g can be i for crude imports/exports
    return cu_imports, cu_exports


def _compute_lower_cycle(country_specific, outflows, recovery_rate, copper_rate, production, forming_yield,
                         forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use,
                         indirect_imports, indirect_exports,
                         max_scrap_share_in_production, forming_scrap, fabrication_scrap):
    ## TODO: Delete this function? Not used, cfg.recycling_strategy == 'econ' doesn't make sense anymore

    lower_cycle_function = None
    if cfg.recycling_strategy == 'base':
        lower_cycle_function = lower_cycle_base_model
    elif cfg.recycling_strategy == 'tramp':
        lower_cycle_function = lower_cycle_tramp_elements
    elif cfg.recycling_strategy == 'econ':
        lower_cycle_function = lower_cycle_econ_tramp
    else:
        raise RuntimeError(f'Recycling strategy {cfg.recycling_strategy} is not defined.')

    return lower_cycle_function(country_specific, outflows, recovery_rate, copper_rate, production, forming_yield,
                                forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use,
                                indirect_imports, indirect_exports,
                                max_scrap_share_in_production, forming_scrap, fabrication_scrap)


def compute_upper_cycle_modelling_approaches():
    production, forming_intermediate, trade, intermediate_fabrication, fabrication_use, indirect_trade, \
    inflows, stocks, outflows = \
        compute_upper_cycle(model_type=cfg.model_type)
    imports, exports = get_imports_and_exports_from_net_trade(trade)
    indirect_imports, indirect_exports = get_imports_and_exports_from_net_trade(indirect_trade)

    return production, forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use, \
           indirect_imports, indirect_exports, inflows, stocks, outflows


def compute_upper_cycle_base_model(country_specific, inflows, outflows, fabrication_yield):
    total_demand = np.sum(inflows, axis=2)

    indirect_imports, indirect_exports = get_indirect_trade(country_specific=country_specific,
                                                            scaler=total_demand,
                                                            inflows=inflows,
                                                            outflows=outflows)
    fabrication_use = inflows - indirect_imports + indirect_exports

    inverse_fabrication_yield = 1 / fabrication_yield
    fabrication_by_category = np.einsum('trgs,g->trgs', fabrication_use, inverse_fabrication_yield)
    forming_fabrication = np.sum(fabrication_by_category, axis=2)

    imports, exports = get_trade(country_specific=country_specific, scaler=total_demand)

    production_plus_trade = forming_fabrication * (1 / cfg.forming_yield)
    production = production_plus_trade + exports - imports

    return production, forming_fabrication, imports, exports, fabrication_use, indirect_imports, indirect_exports, \
           inflows, outflows


def edit_flows(model, iron_production, scrap_in_bof, bof_production, eaf_production, forming_intermediate,
               forming_scrap, intermediate_fabrication, imports, exports, fabrication_use, indirect_imports,
               indirect_exports, reuse, fabrication_scrap, outflows, buffer_eol, buffer_obsolete, scrap_imports,
               scrap_exports, eol_recycling, recycling_scrap, scrap_excess, fabrication_buffer, env_recycling):
    dtesta = scrap_in_bof + iron_production - bof_production
    dtestb = np.all(np.abs(dtesta < 10))
    dtestc = bof_production + eaf_production - np.sum(forming_intermediate, axis=3) - forming_scrap
    dtestd = np.all(np.abs(dtestc < 10))
    dteste = np.sum(intermediate_fabrication, axis=3) - fabrication_scrap - np.sum(fabrication_use, axis=3)
    dtestf = np.all(np.abs(dteste < 10))

    model.get_flowV(ENV_PID, BOF_PID)[:] = iron_production
    model.get_flowV(SCRAP_PID, BOF_PID)[:] = scrap_in_bof
    model.get_flowV(BOF_PID, FORM_PID)[:] = bof_production
    model.get_flowV(SCRAP_PID, EAF_PID)[:] = eaf_production
    model.get_flowV(EAF_PID, FORM_PID)[:] = eaf_production
    model.get_flowV(FORM_PID, IP_PID)[:] = forming_intermediate
    model.get_flowV(FORM_PID, FBUF_PID)[:] = forming_scrap
    model.get_flowV(ENV_PID, IP_PID)[:] = imports
    model.get_flowV(IP_PID, ENV_PID)[:] = exports
    model.get_flowV(IP_PID, FABR_PID)[:] = intermediate_fabrication
    model.get_flowV(FABR_PID, USE_PID)[:] = fabrication_use
    model.get_flowV(USE_PID, BUF_PID)[:] = outflows
    model.get_flowV(FABR_PID, FBUF_PID)[:] = fabrication_scrap
    model.get_flowV(FBUF_PID, SCRAP_PID)[:] = fabrication_buffer
    model.get_flowV(ENV_PID, USE_PID)[:] = indirect_imports
    model.get_flowV(USE_PID, ENV_PID)[:] = indirect_exports
    if reuse is not None:
        model.get_flowV(USE_PID, USE_PID)[:] = reuse
    model.get_flowV(BUF_PID, EOL_PID)[:] = buffer_eol
    model.get_flowV(BUF_PID, OBS_PID)[:] = buffer_obsolete
    model.get_flowV(ENV_PID, EOL_PID)[:] = scrap_imports
    model.get_flowV(EOL_PID, ENV_PID)[:] = scrap_exports
    model.get_flowV(EOL_PID, RECYCLE_PID)[:] = eol_recycling
    model.get_flowV(RECYCLE_PID, SCRAP_PID)[:] = recycling_scrap
    model.get_flowV(SCRAP_PID, EXC_PID)[:] = scrap_excess
    model.get_flowV(ENV_PID, RECYCLE_PID)[:] = env_recycling


def _get_params(model):
    params = model.ParameterDict
    forming_yield = params['Forming_Yield'].Values
    fabrication_yield = params['Fabrication_Yield'].Values
    production_yield = params['Production_Yield'].Values
    recovery_rate = params['Recovery_Rate'].Values
    copper_rate = params['External_Copper_Rate'].Values
    tolerances = params['Copper_Tolerances'].Values
    gi_distribution = params['Good_Intermediate_Distribution'].Values

    return forming_yield, fabrication_yield, production_yield, recovery_rate, copper_rate, tolerances, gi_distribution


def _calc_eaf_share_production(scrap_share):
    eaf_share_production = (scrap_share - cfg.scrap_in_BOF_rate) / (1 - cfg.scrap_in_BOF_rate)
    eaf_share_production = np.maximum(0, eaf_share_production)
    eaf_share_production = np.minimum(1, eaf_share_production)
    return eaf_share_production


def _calc_max_scrap_share(scrap_share_in_production, n_regions):
    max_scrap_share_in_production = np.ones(
        [cfg.n_years, n_regions, cfg.n_scenarios]) * cfg.max_scrap_share_production_base_model
    if scrap_share_in_production is not None:
        max_scrap_share_in_production[cfg.econ_start_index:, :] = scrap_share_in_production
    return max_scrap_share_in_production


def _create_cu_flows(production, fabrication_use, forming_intermediate):
    cu_recycling_scrap = np.zeros_like(fabrication_use)
    cu_scrap_production = np.zeros_like(production)
    cu_production = np.zeros_like(production)
    cu_production_by_intermediate = np.zeros_like(forming_intermediate)
    cu_forming_intermediate = np.zeros_like(forming_intermediate)
    cu_forming_scrap = np.zeros_like(production)
    cu_imports = np.zeros_like(forming_intermediate)
    cu_exports = np.zeros_like(forming_intermediate)
    cu_intermediate_fabrication = np.zeros_like(forming_intermediate)
    cu_fabrication_scrap = np.zeros_like(production)
    cu_fabrication_use = np.zeros_like(fabrication_use)
    cu_indirect_imports = np.zeros_like(fabrication_use)
    cu_indirect_exports = np.zeros_like(fabrication_use)
    cu_inflows = np.zeros_like(fabrication_use)
    cu_stocks = np.zeros_like(fabrication_use)
    cu_outflows = np.zeros_like(fabrication_use)
    cu_buffer = np.zeros_like(fabrication_use)
    cu_fabrication_buffer = np.zeros_like(production)
    cu_buffer_eol = np.zeros_like(fabrication_use)
    cu_scrap_exports = np.zeros_like(fabrication_use)
    cu_scrap_imports = np.zeros_like(fabrication_use)
    cu_eol_recycling = np.zeros_like(fabrication_use)
    cu_available_scrap = np.zeros_like(production)
    cu_tolerated = np.zeros_like(forming_intermediate)
    cu_iron_production = np.zeros_like(production)
    cu_bof_production = np.zeros_like(production)
    cu_eaf_production = np.zeros_like(production)
    cu_scrap_in_bof = np.zeros_like(production)

    return cu_recycling_scrap, cu_scrap_production, cu_production, cu_production_by_intermediate, \
           cu_forming_intermediate, cu_forming_scrap, cu_imports, cu_exports, cu_intermediate_fabrication, \
           cu_fabrication_scrap, cu_fabrication_use, cu_indirect_imports, cu_indirect_exports, \
           cu_inflows, cu_stocks, cu_outflows, cu_buffer, cu_fabrication_buffer, \
           cu_buffer_eol, cu_scrap_exports, cu_scrap_imports, cu_eol_recycling, \
           cu_available_scrap, cu_tolerated, \
           cu_iron_production, cu_bof_production, cu_eaf_production, cu_scrap_in_bof


def compute_stocks(model, inflows, outflows):
    in_use_stock_change = model.get_stock_changeV(USE_PID)
    fabrication_use = model.get_flowV(FABR_PID, USE_PID)
    indirect_imports = model.get_flowV(ENV_PID, USE_PID)
    indirect_exports = model.get_flowV(USE_PID, ENV_PID)
    outflows = model.get_flowV(USE_PID, BUF_PID)
    in_use_stock_change[:] = fabrication_use + indirect_imports - indirect_exports - outflows
    model.calculate_stock_values_from_stock_change(USE_PID)

    use_obsolete = model.get_flowV(BUF_PID, OBS_PID)
    model.get_stock_changeV(OBS_PID)[:] = use_obsolete
    model.calculate_stock_values_from_stock_change(OBS_PID)

    scrap_excess = model.get_flowV(SCRAP_PID, EXC_PID)
    model.get_stock_changeV(EXC_PID)[:] = scrap_excess
    model.calculate_stock_values_from_stock_change(EXC_PID)

    # Buffers

    buf_stock_change = model.get_stock_changeV(BUF_PID)
    buf_stock_change[:] = outflows
    buf_stock_change[1:] -= outflows[:-1]
    model.calculate_stock_values_from_stock_change(BUF_PID)

    total_production_scrap = model.get_flowV(FORM_PID, FBUF_PID) + model.get_flowV(FABR_PID, FBUF_PID)
    f_buf_stock_change = model.get_stock_changeV(FBUF_PID)
    f_buf_stock_change[:] = total_production_scrap
    f_buf_stock_change[1:] -= total_production_scrap[:-1]

    return model


def mass_balance_plausible(main_model):
    """
    Checks if a given mass balance is plausible.
    :return: True if the mass balance for all processes is below 1t of steel, False otherwise.
    """
    balance = main_model.MassBalance()

    balance = np.abs(np.sum(balance, axis=(0, 2)))
    error = balance > 100  # up to 100 t error fine (should be fine across all years, regions, etc.) TODO decide
    if np.any(error):
        error_message = f"Error in mass balance of model\n"
        for idx, error_occured in enumerate(error):
            if idx == 0:  # Environment will always have an error if another process has an error
                continue
            if error_occured:
                error_message += f"\nError in process {idx} '{main_model.ProcessList[idx].Name}': {balance[idx]}"
        error_message += f"\n\nBalance summary: {balance}"
        raise RuntimeError(error_message)
    else:
        return f"Success - Model loaded and checked. \nBalance: {balance}.\n"


def main():
    """
    Recalculates the DMFA dict based on the dynamic stock models and trade_all_areas data.
    Checks the Mass Balance and raises a runtime error if the mass balance is too big.
    Prints success statements otherwise
    :return: None
    """
    load_simson_base_model(country_specific=False, recalculate=True)


if __name__ == "__main__":
    # overwrite config with values given in a config file,
    # if the path to this file is passed as the last argument of the function call.
    if sys.argv[-1].endswith('.yml'):
        cfg.customize(sys.argv[-1])
    main()
