import numpy as np
from src.calc_trade.calc_scrap_trade import get_scrap_trade
from src.tools.config import cfg


def lower_cycle_base_model(country_specific, outflow_buffer, recovery_rate, copper_rate, production, forming_yield,
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
    max_scrap_in_production = production * max_scrap_share_in_production
    available_scrap = np.sum(recycling_scrap, axis=2) + forming_scrap + fabrication_scrap
    scrap_in_production = np.minimum(available_scrap, max_scrap_in_production)

    scrap_share = np.divide(scrap_in_production, production,
                            out=np.zeros_like(scrap_in_production), where=production != 0)
    eaf_share_production = _calc_eaf_share_production(scrap_share)
    eaf_production = production * eaf_share_production
    bof_production = production - eaf_production
    max_scrap_in_bof = cfg.scrap_in_BOF_rate * bof_production
    scrap_in_bof = np.minimum(max_scrap_in_bof, scrap_in_production)
    iron_production = bof_production - scrap_in_bof

    scrap_in_production = scrap_in_bof + eaf_production
    scrap_excess = available_scrap - scrap_in_production

    return iron_production, scrap_in_bof, bof_production, eaf_production, use_eol, use_obsolete, scrap_imports, \
           scrap_exports, eol_recycling, recycling_scrap, scrap_excess


def _calc_eaf_share_production(scrap_share):
    eaf_share_production = (scrap_share - cfg.scrap_in_BOF_rate) / (1 - cfg.scrap_in_BOF_rate)
    eaf_share_production[eaf_share_production <= 0] = 0
    return eaf_share_production
