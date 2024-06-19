import numpy as np
import sympy as sp
from scipy.optimize import bisect, newton
from src.tools.config import cfg
from src.econ_tramp_model.inform_curves import get_average_recov_elasticity, get_average_dis_elasticity, a_average
from src.econ_tramp_model.steel_price_curves import get_bof_prices, get_eaf_prices

# from src.base_model.simson_base_model import
global c2_change_counter
c2_change_counter = 0
global print_messages
print_messages = False
root_newton_numerato_func_changer = 0


def calc_tramp_econ_model(q_st_total, q_eol_g, q_fabrication_buffer, t_eol_share, s_cu_max):
    # load steel price
    exog_eaf_price = get_eaf_prices()
    p_prst_price_from_2023 = get_bof_prices()
    p_prst_price = np.ones(201) * p_prst_price_from_2023[0]
    p_prst_price[123:] = get_bof_prices()

    # account for forming losses to get Q_St -> F3_4 * Frm_yield losses

    # import the stock-driven steel demand for all regions and then adjust it via the
    # TODO: first adjust steel demand via demand elasticity, then determine recovery rates, etc.

    # load parameter

    r_recov_g = np.ones((12, 4, 5)) * 0.8
    s_cu = np.zeros((201, 12, 5))
    q_se_st = np.zeros((201, 12, 5))
    q_eol_total = np.sum(q_eol_g, axis=2)
    for t in range(123, 201):
        if print_messages:
            print(f'\n\n\nCalc year {t + 1900}\n\n\n')
        for r in range(12):
            for s in range(5):
                if print_messages:
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
                if print_messages:
                    print(f'\n\nC2 change counter: {c2_change_counter}\n\n')

    q_pr_st = q_st_total - q_se_st
    if print_messages:
        print(f'\n\nC2 change counter: {c2_change_counter}\n\n')
    return q_pr_st, q_se_st, r_recov_g, s_cu


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


def calc_tramp_econ_model_over_trs(Q_St, Q_Fabrication_Buffer, Q_EoL, S_EoL_net_T, Q_EoL_g, p_prst_price, S_Cu_max,
                                   exog_eaf, S_Cu_alloy_g, check, numerator_change_counter):
    # Constants
    P_0_col = cfg.p_0_col  # 150 #cfg.p_0_col
    P_0_dis = cfg.p_0_dis  # 150 #cfg.p_0_dis
    e_dis = get_average_dis_elasticity()  # -2.548650743789492 #get_average_dis_elasticity()
    e_recov = get_average_recov_elasticity()  # -0.8304820237218398 #get_average_recov_elasticity()
    a = a_average  # 0.25340259071206267 #a_average
    S_reuse = cfg.s_reuse  # cfg.s_reuse #0
    exog_EAF = exog_eaf  # exog_EAF_price #150 #exog_EAF_price
    P_PrSt = p_prst_price  # np.array([100, 50, 75, 175]) # np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p]) #np.array([0.8 * 0.002002, 0.8 * 0.002465341, 0.8 * 0.001822651, 0.8 * 0.001985311]) #np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p])
    S_Cu_0_g = np.array([cfg.s_cu_0_construction, cfg.s_cu_0_machinery, cfg.s_cu_0_products,
                         cfg.s_cu_0_transport])  # np.array([0.003, 0.0025, 0.001, 0.004]) #np.array([cfg.s_cu_0_transport, cfg.s_cu_0_machinery, cfg.s_cu_0_construction, cfg.s_cu_0_products])
    R_0_recov_g = np.array([cfg.r_0_recov_construction, cfg.r_0_recov_machinery, cfg.r_0_recov_products,
                            cfg.r_0_recov_transport])  # np.array([0.9, 0.9, 0.85, 0.5]) #np.array([cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction, cfg.r_0_recov_products])
    S_Cu = sp.symbols('S_Cu')

    # Functions
    def R_recov_g(g_val, P_col):
        return 1 - (1 - R_0_recov_g[g_val]) * ((P_col / P_0_col + a) / (1 + a)) ** e_recov

    # variables for P_col
    c1 = ((np.sum(Q_EoL_g) / np.sum((1 - R_0_recov_g) * Q_EoL_g)) ** (1 / e_recov)) * (1 + a)
    if print_messages:
        print('c1 value: ', c1)
    c2 = (0.8 * S_Cu_max * (Q_St - Q_Fabrication_Buffer)) / (Q_EoL * (1 - S_reuse) * (1 + S_EoL_net_T))
    if print_messages:
        print('c2 value: ', c2)

    # P_col
    def P_col_expr(S_Cu):
        P_col = ((1 - (c2 / S_Cu)) ** (1 / e_recov) * c1 - a) * P_0_col
        return P_col

    # print('P_col: ', P_col_expr(S_Cu))

    # P_dis numerator
    def numerator_P_dis_expr(S_Cu):
        P_col_val = P_col_expr(S_Cu)
        return S_Cu * np.sum([R_recov_g(i, P_col_val) * Q_EoL_g[i] for i in range(4)]) - np.sum(
            [R_recov_g(i, P_col_val) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(4)])

    # print('numerator_P_dis: ', numerator_P_dis_expr(S_Cu))

    # P_dis denominator
    def denominator_P_dis_expr(S_Cu):
        P_col_val = P_col_expr(S_Cu)
        return ((1 / P_0_dis) ** e_dis * np.sum([R_recov_g(i, P_col_val) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(4)]))

    # print('denominator_P_dis: ', denominator_P_dis_expr(S_Cu))

    # P_dis
    def P_dis_expr(S_Cu, P_col):
        numerator_P_dis_expr = S_Cu * np.sum([R_recov_g(i, P_col) * Q_EoL_g[i] for i in range(4)]) - np.sum(
            [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(4)])
        denominator_P_dis_expr = (1 / P_0_dis) ** e_dis * np.sum(
            [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(4)])
        P_dis = (numerator_P_dis_expr / denominator_P_dis_expr) ** (1 / e_dis)
        return P_dis

    # print('P_dis(S_Cu, P_col): ', P_dis_expr(S_Cu, P_col_expr(S_Cu)))

    # function to solve
    def function(S_Cu):
        P_col = P_col_expr(S_Cu)
        P_dis = P_dis_expr(S_Cu, P_col)
        return P_col + P_dis + exog_EAF - P_PrSt

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

    def find_newton_numerator(numerator_P_dis_expr, initial_guess, c2_value, numerator_change_counter):
        if check:
            a = 0
        if c2_value < 0:
            try:
                root_newton_numerator_func = newton(numerator_P_dis_expr, 0.000001)
                if print_messages:
                    print(f"The root of the numerator of P_dis if c2<0 is: {root_newton_numerator_func}")
            except RuntimeError:
                root_newton_numerator_func = 0
                if print_messages:
                    print('No root exists for the numerator with the given parameters!')
        else:
            try:
                if check:
                    a = 0
                root_newton_numerator_func = newton(numerator_P_dis_expr, initial_guess)
                if print_messages:
                    print(f"The root of the numerator of P_dis is: {root_newton_numerator_func}")
            except RuntimeError:
                root_newton_numerator_func = 0
                if print_messages:
                    print('No root exists for the numerator with the given parameters!')

        if root_newton_numerator_func > 1 or root_newton_numerator_func < 0:
            root_newton_numerator_func = 0
            numerator_change_counter += 1
            if print_messages:
                print(f'Root newton numerator func was changed manually due to too high values.'
                      f'\n Change counter is {numerator_change_counter}')

        return root_newton_numerator_func

    root_newton_numerator = find_newton_numerator(numerator_P_dis_expr, initial_guess, c2, numerator_change_counter)
    if print_messages:
        print('root_newton_numerator is: ', root_newton_numerator)

    # Finding root of P_dis_ denominator Newton
    def find_newton_denominator(denominator_P_dis_expr, initial_guess, c2_value):
        if c2_value < 0:
            try:
                root_newton_denominator_func = newton(denominator_P_dis_expr, 0.000001)
                if print_messages:
                    print(f"The root of the denominator of P_dis is: {root_newton_denominator_func}")
            except RuntimeError:
                root_newton_denominator_func = 0
                # print('root_newton_denominator: ', root_newton_denominator_func)
                if print_messages:
                    print(
                        f"No root exists for the denominator with the given parameters and for c2<0, thus root_newton_denominator_func was set to: {root_newton_denominator_func}")
        else:
            try:
                root_newton_denominator_func = newton(denominator_P_dis_expr, initial_guess)
                if print_messages:
                    print(f"The root of the denominator of P_dis is: {root_newton_denominator_func}")
            except RuntimeError:
                root_newton_denominator_func = 0
                if print_messages:
                    print(
                        f"No root exists for the denominator with the given parameters, thus root_newton_denominator_func was set to: {root_newton_denominator_func}")
        return root_newton_denominator_func

    root_newton_denominator = find_newton_denominator(denominator_P_dis_expr, initial_guess, c2)
    if print_messages:
        print(f"The root of the denominator of P_dis is: {root_newton_denominator}")

    def decide_starting_values_for_bisec(root_newton_numerator, root_newton_denominator, c2_value_exceeded, c2_value):
        if c2_value < 0:
            start_value_bisec_func = root_newton_numerator * 1.000001  # + 0.00000000000001236580800
            end_value_bisec_func = 0.5
            if print_messages:
                print(
                    f"The root of the denominator does not exist for c2<0 ({c2}), thus the new starting value for bisec is the root of the numerator: {start_value_bisec_func}.",
                    f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_numerator == 0:
            start_value_bisec_func = c2_value_exceeded
            end_value_bisec_func = root_newton_denominator * 0.99999
            if print_messages:
                print(
                    f"The root of the numerator is zero, thus the new starting value for bisec is the value of c2*1.00001: {start_value_bisec_func}.",
                    f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_denominator > root_newton_numerator:
            start_value_bisec_func = root_newton_numerator * 1.000001
            end_value_bisec_func = root_newton_denominator * 0.99999  # Frage, ob dieser Weert wirklich kleiner sein sollte als seine gefundene root
            if print_messages:
                print(
                    f"The root of the numerator is smaller than the root of the denominator, thus the new starting value for bisec is: {start_value_bisec_func}.",
                    f"And the new end value for bisec is: {end_value_bisec_func}")
        elif root_newton_denominator < root_newton_numerator:
            start_value_bisec_func = root_newton_denominator * 1.00001
            end_value_bisec_func = root_newton_numerator * 0.99999
            if print_messages:
                print(
                    f"The root of the denominator is smaller than the root of the numerator, thus the new starting value for bisec is: {start_value_bisec_func}.",
                    f"And the new end value for bisec is: {end_value_bisec_func}")
        else:
            print("Error during root finding")
        return start_value_bisec_func, end_value_bisec_func

    start_value_bisec, end_value_bisec = decide_starting_values_for_bisec(root_newton_numerator,
                                                                          root_newton_denominator, initial_guess, c2)
    if print_messages:
        print('The new starting value for bisec is: ', start_value_bisec)
        print('The new end value for bisec is: ', end_value_bisec)
    # print(f"numerator value for S_Cu = {} ", start_value_bisec)

    # print('The initial_guess_plot is: ', initial_guess_plot)

    # array = np.linspace(0.0004283753649407969, 0.05, 10000)

    # Plots of P_dis, P_dis_numerator, P_dis_denominator
    # Generate a range of S_Cu values for plotting numerator and denominator
    # S_Cu_values_numerator = np.linspace(0.000001, 0.1, 500)
    # numerator_values = [numerator_P_dis_expr(S_Cu) for S_Cu in S_Cu_values_numerator]

    # Generate a range of S_Cu values for plotting numerator and denominator
    # S_Cu_values_denominator = np.linspace(0.000001, 0.1, 500)
    # denominator_values = [denominator_P_dis_expr(S_Cu) for S_Cu in S_Cu_values_denominator]

    # Generate a range of S_Cu values for plotting P_dis
    # S_Cu_values_p_dis = np.linspace(0.000001, 0.3, 100)
    # P_dis_values = [P_dis_expr(S_Cu, P_col_expr(S_Cu_max)) for S_Cu in S_Cu_values_p_dis]

    # Generate a range of S_Cu values for plotting P_col
    # S_Cu_values_p_col = np.linspace(0.000001, 0.1, 100)
    # P_col_values = [P_col_expr(S_Cu) for S_Cu in S_Cu_values_p_col]

    # Plot R_recov_g functions
    # Values for P_col
    # P_col_values = np.linspace(1, 1000, 100)

    if print_messages:
        print('start_value_bisec: ', start_value_bisec)
        print('end_value_bisec: ', end_value_bisec)

    # Usage
    S_Cu_bisection = find_root_bisection(start_value_bisec, end_value_bisec)
    S_Cu_newton = find_root_newton(S_Cu_bisection)

    if print_messages:
        print('\n')
        print(f"Root found by bisection method: {S_Cu_bisection}")
        print(f"Root found by Newton-Raphson method: {S_Cu_newton}")

    P_col = P_col_expr(S_Cu_newton)
    P_dis = P_dis_expr(S_Cu_newton, P_col)

    R_recov_c = R_recov_g(0, P_col)
    R_recov_m = R_recov_g(1, P_col)
    R_recov_p = R_recov_g(2, P_col)
    R_recov_t = R_recov_g(3, P_col)

    r_recov_g = np.array([R_recov_c, R_recov_m, R_recov_p, R_recov_t])

    if R_recov_p < 0:
        R_recov_p = 0

    P_dis_col_eaf = P_col + P_dis + exog_EAF
    price_diff = P_PrSt - P_dis_col_eaf

    q_se_st = (0.8 * S_Cu_max * (Q_St - Q_Fabrication_Buffer)) / S_Cu_newton
    # if q_se_st < 0 or q_se_st > Q_EoL:
    #    raise RuntimeError('Quantity of q_se_st is negative or greater than Q_EoL!')

    if print_messages:
        print('Quantitiy of q_se_st is: ', q_se_st)
        print(f"P_col: {P_col}")
        print(f"P_dis: {P_dis}")
        print(f"P_col + P_dis + exog(EAF) : {P_dis_col_eaf}")
        print(f"P_Se_St - P_PrSt: {price_diff}")
        print(f"R_recov transport: {R_recov_t}")
        print(f"R_recov machinery: {R_recov_m}")
        print(f"R_recov construction: {R_recov_c}")
        print(f"R_recov products: {R_recov_p}")

    return r_recov_g, S_Cu_newton, q_se_st


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
