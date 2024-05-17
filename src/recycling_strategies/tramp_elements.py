import numpy as np
from src.calc_trade.calc_scrap_trade import get_scrap_trade


def lower_cycle_tramp_elements(country_specific, outflow_buffer, recovery_rate, copper_rate, production, forming_yield,
                               forming_intermediate, imports, exports, intermediate_fabrication, fabrication_use,
                               indirect_imports, indirect_exports,
                               max_scrap_share_in_production, forming_scrap, fabrication_scrap):
    use_eol = np.einsum('trgs,g->trgs', outflow_buffer, recovery_rate)
    use_obsolete = outflow_buffer - use_eol
    scrap_imports, scrap_exports = get_scrap_trade(country_specific=country_specific,
                                                   available_scrap_by_category=use_eol)

    total_eol_scrap = use_eol + scrap_imports - scrap_exports
    scrap_is_positive = np.all(total_eol_scrap >= 0)
    if not scrap_is_positive:
        raise RuntimeError('Scrap is not positive')

    eol_recycling = total_eol_scrap
    recycling_scrap = total_eol_scrap
    cu_external = np.einsum('trgs,g->trgs', eol_recycling, copper_rate)

    production_by_intermediate = np.einsum('tris,i->tris', forming_intermediate, 1 / forming_yield)

    cu_recycling_scrap, cu_scrap_production, cu_production, cu_production_by_intermediate, \
    cu_forming_intermediate, cu_forming_scrap, cu_imports, cu_exports, cu_forming_fabrication, \
    cu_fabrication_scrap, cu_fabrication_use, cu_indirect_imports, cu_indirect_exports, \
    cu_inflows, cu_stocks, cu_outflows, cu_buffer, \
    cu_buffer_eol, cu_scrap_exports, cu_scrap_imports, cu_eol_recycling = \
        _create_cu_flows(production, fabrication_use, forming_intermediate)

    for t in range(1, 201):
        a = 0

    return iron_production, scrap_in_bof, bof_production, eaf_production, use_eol, use_obsolete, scrap_imports, \
           scrap_exports, eol_recycling, recycling_scrap, scrap_excess


def _create_cu_flows(production, fabrication_use, forming_intermediate):
    cu_recycling_scrap = np.zeros_like(fabrication_use)
    cu_scrap_production = np.zeros_like(production)
    cu_production = np.zeros_like(production)
    cu_production_by_intermediate = np.zeros_like(forming_intermediate)
    cu_forming_intermediate = np.zeros_like(forming_intermediate)
    cu_forming_scrap = np.zeros_like(forming_intermediate)
    cu_imports = np.zeros_like(forming_intermediate)
    cu_exports = np.zeros_like(forming_intermediate)
    cu_forming_fabrication = np.zeros_like(forming_intermediate)
    cu_fabrication_scrap = np.zeros_like(fabrication_use)
    cu_fabrication_use = np.zeros_like(fabrication_use)
    cu_indirect_imports = np.zeros_like(fabrication_use)
    cu_indirect_exports = np.zeros_like(fabrication_use)
    cu_inflows = np.zeros_like(fabrication_use)
    cu_stocks = np.zeros_like(fabrication_use)
    cu_outflows = np.zeros_like(fabrication_use)
    cu_buffer = np.zeros_like(fabrication_use)
    cu_buffer_eol = np.zeros_like(fabrication_use)
    cu_scrap_exports = np.zeros_like(fabrication_use)
    cu_scrap_imports = np.zeros_like(fabrication_use)
    cu_eol_recycling = np.zeros_like(fabrication_use)

    return cu_recycling_scrap, cu_scrap_production, cu_production, cu_production_by_intermediate, \
           cu_forming_intermediate, cu_forming_scrap, cu_imports, cu_exports, cu_forming_fabrication, \
           cu_fabrication_scrap, cu_fabrication_use, cu_indirect_imports, cu_indirect_exports, \
           cu_inflows, cu_stocks, cu_outflows, cu_buffer, \
           cu_buffer_eol, cu_scrap_exports, cu_scrap_imports, cu_eol_recycling
