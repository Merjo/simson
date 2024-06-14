import numpy as np
from matplotlib import pyplot as plt
from src.read_data.load_data import load_region_names_list
from src.tools.config import cfg
from src.modelling_approaches.model_3_change_driven import get_change_driven_model_past_upper_cycle
from src.modelling_approaches.model_2_stock_driven import get_stock_driven_model_past_upper_cycle
from src.modelling_approaches.model_1_inflow_driven import get_inflow_driven_model_past_upper_cycle
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data
from src.calc_trade.calc_trade_tools import scale_trade
from src.base_model.load_params import get_cullen_fabrication_yield, get_cullen_forming_yield, \
    get_daehn_good_intermediate_distribution


def compute_upper_cycle(model_type=cfg.model_type, country_specific=False):  # decide country_specific
    # TODO: include REUSE ???

    if model_type == 'change':
        get_upper_cycle_function = get_change_driven_model_past_upper_cycle
    elif model_type == 'stock':
        get_upper_cycle_function = get_stock_driven_model_past_upper_cycle
    elif model_type == 'inflow':
        get_upper_cycle_function = get_inflow_driven_model_past_upper_cycle
    past_production, past_forming_intermediate, past_trade, past_intermediate_fabrication, past_fabrication_use, past_indirect_trade, past_inflows, \
    past_stocks, past_outflows = \
        get_upper_cycle_function(country_specific=country_specific)

    past_production = _add_scenario_dimension(past_production)
    past_trade = _add_scenario_dimension(past_trade)
    past_intermediate_fabrication = _add_scenario_dimension(past_intermediate_fabrication)
    past_fabrication_use = _add_scenario_dimension(past_fabrication_use)
    past_indirect_trade = _add_scenario_dimension(past_indirect_trade)

    do_econ_model = cfg.do_model_economy
    future_dsms = load_model_dsms(country_specific=country_specific,
                                  do_past_not_future=False,
                                  model_type=model_type,
                                  do_econ_model=do_econ_model,
                                  # TODO: decide whether to implement here or in econ model
                                  recalculate=False)  # TODO: change, only leave for now

    inflows, stocks, outflows = get_dsm_data(future_dsms)

    future_indirect_trade = scale_trade(past_indirect_trade, scaler=inflows[123 - 1:],
                                        do_past_not_future=False)
    indirect_trade = np.concatenate((past_indirect_trade, future_indirect_trade), axis=0)

    fabrication_yield = np.array(get_cullen_fabrication_yield())
    gi_distribution = get_daehn_good_intermediate_distribution()
    forming_yield = get_cullen_forming_yield()

    fabrication_use = inflows - indirect_trade
    intermediate_fabrication = np.einsum('trgs,g,gi->tris', fabrication_use, 1 / fabrication_yield, gi_distribution)

    future_trade = scale_trade(past_trade, scaler=intermediate_fabrication[123 - 1:],
                               # de facto the scaler is inflows, this just helps because it has the same dimensions
                               do_past_not_future=False)
    trade = np.concatenate((past_trade, future_trade), axis=0)

    forming_intermediate = intermediate_fabrication - trade
    production = np.einsum('tris,i->trs', forming_intermediate, 1 / forming_yield)

    # test if something went wrong during model approaches
    test_a = np.all(fabrication_use[:123] - past_fabrication_use < 4)
    test_BBBB = np.abs(fabrication_use[:123] - past_fabrication_use)
    test_AAA = np.abs(fabrication_use[:123] - past_fabrication_use) < 4
    test_b = np.all(np.abs(intermediate_fabrication[:123, :, :, 1] - past_intermediate_fabrication[:, :, :, 1]) < 4)
    test_c = np.all(np.abs(production[1:123, :, 1] - past_production[1:, :, 1]) < 4)
    if not (test_a and test_b and test_c):
        raise RuntimeError('Something went wrong during model approach calculation.')

    # production might have been changed in the first year depending on the model approach
    production[:123] = past_production

    return production, forming_intermediate, trade, intermediate_fabrication, fabrication_use, indirect_trade, \
           inflows, stocks, outflows


def _add_scenario_dimension(data):
    new_shape = data.shape + (cfg.n_scenarios,)
    data = np.expand_dims(data, axis=-1)
    data = np.broadcast_to(data, new_shape)
    return data


def test(model_type=cfg.model_type, visualise=True):
    production, forming_intermediate, trade, intermediate_fabrication, fabrication_use, indirect_trade, \
    inflows, stocks, outflows = compute_upper_cycle(model_type)

    print(f'{model_type.capitalize()}-based model upper cycle loading succesful. \n'
          f'\nCheck shapes\n')
    print(f'Production shape: {production.shape}')
    print(f'Trade shape: {trade.shape}')
    print(f'Forming-Fabrication shape: {intermediate_fabrication.shape}')
    print(f'Fabrication-Use shape: {fabrication_use.shape}')
    print(f'Indirect trade shape: {indirect_trade.shape}')
    print(f'Inflow shape: {inflows.shape}')
    print(f'Stocks shape: {stocks.shape}')
    print(f'Outflows shape: {outflows.shape}')

    if visualise:
        _vis_production(production)


def _vis_production(production):
    production = production[:, :, 1]  # choose SSP2
    regions = load_region_names_list()

    years = range(1900, 2101)
    for r, region in enumerate(regions):
        plt.plot(years, production[:, r])
    plt.xlabel('Years')
    plt.ylabel('Production (t)')
    plt.legend(regions)
    plt.title('Production calculated by Upper Cycle function')
    plt.show()


if __name__ == '__main__':
    test(model_type=cfg.model_type, visualise=True)
