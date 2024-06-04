from os.path import join
import yaml
import numpy as np


class Config:
    """
    Class of SIMSON configurations. Contains both general configurations like SSP scenarios
    and product categories to use as well as adaptable parameters like steel price change.
    A common instance of this class ('cfg') is used by most other files, its attributes can be
    adapted with the 'customize' method.
    """

    def __init__(self):
        """
        Creates instance of config class with default parameters. These can :func:`print`
        modified through :py:func:`src.tools.config.Config#customize` method.
        """
        self.s_reuse = 0

        # parameters for economic cost curve informing
        self.p_0_col = 150
        self.p_0_dis = 150
        self.price_scenario = '1_5_degree'  # options: baseline, 1_5_degree

        '''Curve Informing Recovery Rate'''

        # recovery rate transport
        self.r_0_recov_transport = 0.9
        self.rp_transport = 0.95
        self.p0_transport = 150
        self.pp_transport = 300
        self.r_free_transport = 0

        # recovery rate machinery
        self.r_0_recov_machinery = 0.9
        self.rp_machinery = 0.95
        self.p0_machinery = 150
        self.pp_machinery = 300
        self.r_free_machinery = 0

        # recovery rate construction
        self.r_0_recov_construction = 0.85
        self.rp_construction = 0.9
        self.p0_construction = 150
        self.pp_construction = 300
        self.r_free_construction = 0

        # recovery rate products
        self.r_0_recov_products = 0.5
        self.rp_products = 0.7
        self.p0_products = 150
        self.pp_products = 300
        self.r_free_products = 0

        '''Curve Informing Copper Removal'''
        # copper removal transport
        self.s_cu_0_transport = 0.003
        self.s_cu_p_transport = 0.0004

        # copper removal machinery
        self.s_cu_0_machinery = 0.0025
        self.s_cu_p_machinery = 0.0004

        # copper removal construction
        self.s_cu_0_construction = 0.001
        self.s_cu_p_construction = 0.0004

        # copper removal products
        self.s_cu_0_products = 0.004
        self.s_cu_p_products = 0.0004

        # S_Cu_slloy_max values are known as tolerances are always met by being a factor of 0.8 under their max values
        self.s_cu_alloy_t = 0.8 * 0.002002
        self.s_cu_alloy_m = 0.8 * 0.002465341
        self.s_cu_alloy_c = 0.8 * 0.001822651
        self.s_cu_alloy_p = 0.8 * 0.001985311


cfg = Config()

if __name__ == '__main__':
    cfg.generate_yml()
