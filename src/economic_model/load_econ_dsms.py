from src.odym_extension.MultiDim_DynamicStockModel import MultiDim_DynamicStockModel
from src.base_model.load_dsms import load_dsms
from src.tools.config import cfg


def load_econ_dsms(country_specific, p_st, p_0_st, recalculate):
    dsms = load_dsms(country_specific, recalculate)
    factor = (p_st / p_0_st) ** cfg.elasticity_steel
    for region_dsms in dsms:
        for category_dsms in region_dsms:
            for scenario_idx, scenario_dsm in enumerate(category_dsms):
                scenario_dsm.i[cfg.econ_base_year - cfg.start_year + 1:] *= factor[:, scenario_idx]
                new_scenarion_dsm = MultiDim_DynamicStockModel(t=scenario_dsm.t,
                                                               i=scenario_dsm.i,
                                                               lt=scenario_dsm.lt)
                new_scenarion_dsm.compute_all_inflow_driven()
                scenario_dsm.copy_dsm_values(new_scenarion_dsm)
    return dsms


def _test():
    load_econ_dsms(country_specific=False, p_st=10, p_0_st=5, recalculate=True)


if __name__ == '__main__':
    _test()
