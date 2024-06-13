import numpy as np
import sympy as sp
from scipy.optimize import bisect, newton
from src.tools.config import cfg
from src.econ_tramp_model.inform_curves import get_average_recov_elasticity, get_average_dis_elasticity, a_average
from src.econ_tramp_model.steel_price_curves import get_bof_prices, get_eaf_prices

# from src.base_model.simson_base_model import
global c2_change_counter
c2_change_counter = 0


def calc_tramp_econ_model(q_st_total, q_eol_g, q_fabrication_buffer, t_eol_share, p_0_st, p_st, r_0_recov_g, ip_tlrc_i,
                          s_cu_alloy_g, s_cu_max):
    # load steel price
    exog_eaf_price = get_eaf_prices()
    p_prst_price_from_2023 = get_bof_prices()
    p_prst_price = np.ones(201) * p_prst_price_from_2023[0]
    p_prst_price[123:] = get_bof_prices()

    # account for forming losses to get Q_St -> F3_4 * Frm_yield losses

    # import the stock-driven steel demand for all regions and then adjust it via the
    # TODO: first adjust steel demand via demand elasticity, then determine recovery rates, etc.

    # load parameter

    r_recov_g = np.ones((201, 12, 4, 5)) * 0.8
    s_cu = np.zeros((201, 12, 5))
    q_se_st = np.zeros((201, 12, 5))
    q_eol_total = np.sum(q_eol_g, axis=2)
    for t in range(123, 201):
        print(f'\n\n\nCalc year {t + 1900}\n\n\n')
        for r in range(12):
            for s in range(5):
                print(f'\n\n\nCalc year {t + 1900}, region {r}, scenario {s}\n\n\n')
                trs_r_recov, trs_s_cu, trs_q_se_st = calc_tramp_econ_model_over_trs(q_st_total[t, r, s],
                                                                                    q_fabrication_buffer[t, r, s],
                                                                                    q_eol_total[t, r, s],
                                                                                    t_eol_share[t, r, s],
                                                                                    q_eol_g[t, r, :, s],
                                                                                    p_prst_price[t],
                                                                                    s_cu_max[t, r, s],
                                                                                    cfg.exog_eaf_USD98)  # exog_eaf_price[0])
                r_recov_g[t, r, :, s] = trs_r_recov
                s_cu[t, r, s] = trs_s_cu
                q_se_st[t, r, s] = trs_q_se_st
                print(f'\n\nC2 change counter: {c2_change_counter}\n\n')

    q_pr_st = q_st_total - q_se_st
    print(f'\n\nC2 change counter: {c2_change_counter}\n\n')
    return q_pr_st, q_se_st, r_recov_g, s_cu


def calc_tramp_econ_model_over_trs(Q_St, Q_Fabrication_Buffer, Q_EoL, S_EoL_net_T, Q_EoL_g, p_prst_price, S_Cu_max,
                                   exog_eaf):
    if (Q_EoL_g == 0).any():
        return np.zeros_like(Q_EoL), np.zeros_like(Q_EoL), np.zeros_like(Q_EoL)

    P_0_col = cfg.p_0_col
    P_0_dis = cfg.p_0_dis
    e_dis = get_average_dis_elasticity()
    e_recov = get_average_recov_elasticity()
    a = a_average
    # S_Cu_max = 0.0004  # 0.0004 is the minimum value S_Cu_max can obtain as lowest tolerance is 0.04wt%
    S_reuse = cfg.s_reuse
    p_prst = p_prst_price
    S_Cu_alloy_g = np.array([cfg.s_cu_alloy_c, cfg.s_cu_alloy_m, cfg.s_cu_alloy_p,
                             cfg.s_cu_alloy_t])  # np.array([0.8 * 0.002002, 0.8 * 0.002465341, 0.8 * 0.001822651, 0.8 * 0.001985311]) #np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p])
    S_Cu_0_g = np.array([cfg.s_cu_0_construction, cfg.s_cu_0_machinery, cfg.s_cu_0_products,
                         cfg.s_cu_0_transport])  # np.array([0.003, 0.0025, 0.001, 0.004]) #np.array([cfg.s_cu_0_transport, cfg.s_cu_0_machinery, cfg.s_cu_0_construction, cfg.s_cu_0_products])
    R_0_recov_g = np.array([cfg.r_0_recov_construction, cfg.r_0_recov_machinery, cfg.r_0_recov_products,
                            cfg.r_0_recov_transport])  # np.array([0.9, 0.9, 0.85, 0.5]) #np.array([cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction, cfg.r_0_recov_products])
    S_Cu = sp.symbols('S_Cu')

    # Defining functions
    # Recovery rate per good
    def R_recov_g(g_val, P_col):
        return 1 - (1 - R_0_recov_g[g_val]) * ((P_col / P_0_col + a) / (1 + a)) ** e_recov

    # variables for P_col
    c1 = ((np.sum(Q_EoL_g) / np.sum((1 - R_0_recov_g) * Q_EoL_g)) ** (1 / e_recov)) * (1 + a)

    c2 = (0.8 * S_Cu_max * (Q_St - Q_Fabrication_Buffer)) / (Q_EoL * (1 - S_reuse) * (1 + S_EoL_net_T))
    if abs(c2) < 0.00015:
        print(f'C2 value was too close to zero ({c2})')
        if c2 < 0:
            c2 = -0.00014
        else:
            c2 = 0.00015
        print(f'\n\n\nHence c2 was changed the {c2_change_counter} time. \n\n\n')
        c2_change_counter += 1

    print('c2 value: ', c2)

    # P_col
    def P_col_expr(S_Cu):
        P_col = ((1 - c2 / S_Cu) ** (1 / e_recov) * c1 - a) * P_0_col
        return P_col

    # P_dis numerator
    def numerator_P_dis_expr(S_Cu):
        P_col_val = P_col_expr(S_Cu)
        return S_Cu * np.sum([R_recov_g(i, P_col_val) * Q_EoL_g[i] for i in range(4)]) - np.sum(
            [R_recov_g(i, P_col_val) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(4)])

    # P_dis denominator
    def denominator_P_dis_expr(S_Cu):
        P_col_val = P_col_expr(S_Cu)
        return ((1 / P_0_dis) ** e_dis * np.sum([R_recov_g(i, P_col_val) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(4)]))

    # P_dis
    def P_dis_expr(S_Cu, P_col):
        numerator_P_dis_expr = S_Cu * np.sum([R_recov_g(i, P_col) * Q_EoL_g[i] for i in range(4)]) - np.sum(
            [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(4)])
        denominator_P_dis_expr = (1 / P_0_dis) ** e_dis * np.sum(
            [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(4)])
        P_dis = (numerator_P_dis_expr / denominator_P_dis_expr) ** (1 / e_dis)
        return P_dis

    # function to solve
    def function(S_Cu):
        P_col = P_col_expr(S_Cu)
        P_dis = P_dis_expr(S_Cu, P_col)
        return P_col + P_dis + exog_eaf - p_prst

    # derivative of function to solve
    def derivative(S_Cu, epsilon=1e-6):
        return (function(S_Cu + epsilon) - function(S_Cu - epsilon)) / (2 * epsilon)

    # Root finding methods
    def find_root_bisection(a, b, tol=1e-6):
        return bisect(function, a, b, xtol=tol)

    def find_root_newton(initial_guess, tol=1e-6):
        return newton(function, initial_guess, fprime=derivative, tol=tol)

    # Finding root of P_dis_ numerator Newton
    initial_guess = c2 * 1.0000001

    def find_newton_numerator(numerator_P_dis_expr, initial_guess, c2_value):
        if c2_value < 0:
            try:
                root_newton_numerator_func = newton(numerator_P_dis_expr, 0.000001) * 1.00001
                print(f"The root of the numerator of P_dis if c2<0 is: {root_newton_numerator_func}")
            except RuntimeError:
                root_newton_numerator_func = 0
                print('No root exists for the numerator with the given parameters!')
        else:
            try:
                root_newton_numerator_func = newton(numerator_P_dis_expr, initial_guess) * 1.00001
                print(f"The root of the numerator of P_dis is: {root_newton_numerator_func}")
            except RuntimeError:
                root_newton_numerator_func = 0
                print('No root exists for the numerator with the given parameters!')
        return root_newton_numerator_func

    root_newton_numerator = find_newton_numerator(numerator_P_dis_expr, initial_guess, c2)

    # Finding root of P_dis_ denominator Newton
    def find_newton_denominator(denominator_P_dis_expr, initial_guess, c2_value):
        if c2_value < 0:
            try:
                root_newton_denominator_func = newton(denominator_P_dis_expr, 0.000001)
                print(f"The root of the denominator of P_dis is: {root_newton_denominator_func}")
            except RuntimeError:
                root_newton_denominator_func = 0
                print('root_newton_denominator: ', root_newton_denominator_func)
                print(
                    f"No root exists for the denominator with the given parameters and for c2<0, thus root_newton_denominator_func was set to: {root_newton_denominator_func}")
        else:
            try:
                root_newton_denominator_func = newton(denominator_P_dis_expr, initial_guess) * 1.00001
                print(f"The root of the denominator of P_dis is: {root_newton_denominator_func}")
            except RuntimeError:
                root_newton_denominator_func = 0
                print(
                    f"No root exists for the denominator with the given parameters, thus root_newton_denominator_func was set to: {root_newton_denominator_func}")
        return root_newton_denominator_func

    root_newton_denominator = find_newton_denominator(denominator_P_dis_expr, initial_guess, c2)

    print(f"The root of the denominator of P_dis is: {root_newton_denominator}")

    def decide_starting_values_for_bisec(root_newton_numerator, root_newton_denominator, c2_value_exceeded, c2_value):
        if c2_value < 0:
            start_value_bisec_func = root_newton_numerator
            end_value_bisec_func = 0.5
            print(
                f"The root of the denominator does not exist for c2<0 ({c2}), thus the new starting value for bisec is the root of the numerator: {start_value_bisec_func}.",
                f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_numerator == 0:
            start_value_bisec_func = c2_value_exceeded
            end_value_bisec_func = root_newton_denominator
            print(
                f"The root of the numerator is zero, thus the new starting value for bisec is the value of c2*1.00001: {start_value_bisec_func}.",
                f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_denominator > root_newton_numerator:
            start_value_bisec_func = root_newton_numerator
            end_value_bisec_func = root_newton_denominator
            print(
                f"The root of the numerator is smaller than the root of the denominator, thus the new starting value for bisec is: {start_value_bisec_func}.",
                f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_denominator < root_newton_numerator:
            root_newton_denominator = root_newton_denominator * (
                    1 / 0.99999) * 1.00001  # inital adjustments need to be reversed
            root_newton_numerator = root_newton_numerator * (
                    1 / 1.00001) * 0.99999  # inital adjustments need to be reversed
            start_value_bisec_func = root_newton_denominator
            end_value_bisec_func = root_newton_numerator
            print(
                f"The root of the denominator is smaller than the root of the numerator, thus the new starting value for bisec is: {start_value_bisec_func}.",
                f"And the new end value for bisec is: {end_value_bisec_func}")
        else:
            print("Error during root finding")
        return start_value_bisec_func, end_value_bisec_func

    start_value_bisec, end_value_bisec = decide_starting_values_for_bisec(root_newton_numerator,
                                                                          root_newton_denominator, initial_guess, c2)

    # solve econ model
    s_cu_bisection = find_root_bisection(start_value_bisec, end_value_bisec)
    s_cu_newton = find_root_newton(s_cu_bisection)

    if np.any(s_cu_newton <= 0) or np.any(s_cu_newton >= 1):
        raise ValueError(f's_cu_newton wrong: {s_cu_newton}')

    print('\n')
    print(f"Root found by bisection method: {s_cu_bisection}")
    print(f"Root found by Newton-Raphson method: {s_cu_newton}")

    p_col = P_col_expr(s_cu_newton)
    p_dis = P_dis_expr(s_cu_newton, p_col)

    r_recov_c = R_recov_g(0, p_col)
    r_recov_m = R_recov_g(1, p_col)
    r_recov_p = R_recov_g(2, p_col)
    r_recov_t = R_recov_g(3, p_col)

    r_recov_g = np.array([r_recov_c, r_recov_m, r_recov_p, r_recov_t])

    r_recov_g = np.maximum(0, r_recov_g)

    q_se_st = (0.8 * S_Cu_max * (Q_St - Q_Fabrication_Buffer)) / s_cu_newton

    q_se_st = np.maximum(0, q_se_st)

    return r_recov_g, s_cu_newton, q_se_st

    p_sum_dis_col_eaf = p_col + p_dis + exog_eaf
    price_diff_Pr_Se = p_prst - p_sum_dis_col_eaf

    print(f"P_col: {p_col}")
    print(f"P_dis: {p_dis}")
    print(f"P_col + P_dis + exog(EAF) : {p_sum_dis_col_eaf}")
    print(f"P_Se_St - P_PrSt: {price_diff_Pr_Se}")
    print(f"R_recov transport: {r_recov_t}")
    print(f"R_recov machinery: {r_recov_m}")
    print(f"R_recov construction: {r_recov_c}")
    print(f"R_recov products: {r_recov_p}")


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
