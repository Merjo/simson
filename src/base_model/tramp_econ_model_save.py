import numpy as np
from src.tools.config import cfg


def calc_tramp_econ_model(q_st, q_eol_g, t_eol_g, p_0_st, p_st, r_0_recov_g, ip_tlrc_i, s_cu_alloy_g):
    p_0_col = None  # TODO: needs to be loaded
    p_0_dis = None  # TODO: needs to be loaded
    exog_eaf = cfg.exog_eaf_USD98  # TODO: Needs to be checked for correctness
    a_recov = _get_a_recov(r_0_recov_g)

    ### TODO: econ model solving
    q_pr_st = q_st * 0.4
    q_se_st = q_st * 0.2
    r_recov_g = np.ones((12, 4)) * 0.8
    s_cu_g = np.ones((12, 4)) * 0.001
    return q_pr_st, q_se_st, r_recov_g, s_cu_g


def _get_a_recov(initial_recovery_rate):
    a_recov = 1 / (((1 - initial_recovery_rate) / (1 - cfg.r_free_recov)) ** (
            1 / cfg.elasticity_scrap_recovery_rate) - 1)
    if np.any(a_recov < 0):
        _warn_too_high_r_free('scrap recovery rate')
    return np.maximum(0, a_recov)


def _warn_too_high_r_free(type: str):
    message = f'R_free was partly higher than initial {type}. Hence a of {type} was made positive, indirectly ' \
              f'changing r_free to be equal to the initial {type} in cases where it was greater.'
    raise RuntimeWarning(message)


def _test():
    q_st = np.random.rand(201, 12, 5) * 40000 + 40000
    q_eol_g = np.random.rand(201, 12, 4, 5) * 5000 + 5000
    t_eol_g = np.random.rand(201, 12, 4, 5) * 4000 - 2000
    s_cu_alloy_g = np.array([0.002835, 0.001419, 0.001771, 0.001209])
    p_st = 840
    p_0_st = 800
    r_0_recov_g = np.random.rand(4)
    ip_tlrc_i = np.random.rand(19) / 5
    q_pr_st, q_se_st, r_recov_g, s_cu_g = \
        calc_tramp_econ_model(q_st, q_eol_g, t_eol_g, p_0_st, p_st, r_0_recov_g, ip_tlrc_i, s_cu_alloy_g)

    print(f'Primary Steel: \n {q_pr_st}\n\n')
    print(f'Secondary Steel: \n {q_se_st}\n\n')
    print(f'Recovery Rate: \n {r_recov_g}\n\n')
    print(f'Copper share: \n {s_cu_g}\n\n')


if '__name__' == '__main__':
    _test()
