import numpy as np
from src.tools.config import cfg


def calc_tramp_econ_model_one_year(q_st, q_eol_g, t_eol_g, p_0_st, p_st, r_0_recov_g, ip_tlrc_i):
    p_0_col = None  # TODO: needs to be loaded
    p_0_dis = None  # TODO: needs to be loaded
    exog_eaf = cfg.exog_eaf_USD98  # TODO: Needs to be checked for correctness
    a_recov = _get_a_recov(r_0_recov_g)

    ### TODO: econ model solving
    q_pr_st = None
    q_se_st = None
    r_recov_g = None
    s_cu_g = None
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
