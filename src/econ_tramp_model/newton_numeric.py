import numpy as np
from scipy.optimize import bisect, newton
import sys
import os

# Add the specific directory to the Python path
script_directory1 = '/Users/marcelgeller/PycharmProjects/curve_informing_simson/venv/Testings/curve_informing'
# sys.path.append(script_directory1)
from config import cfg

# Import e_recov from plot_all_curves_informed.py
from inform_curves import get_average_recov_elasticity, get_average_dis_elasticity, get_parameters_a_for_recov_curves, \
    a_average
from steel_price_curves import get_price_for_scenario_and_year

# defined year
t_price = 2050

# Call the price function to get prices
exog_EAF_price, P_PrSt_price = get_price_for_scenario_and_year(t_price)

# Constants
P_0_col = cfg.p_0_col
P_0_dis = cfg.p_0_dis
e_dis = get_average_dis_elasticity()
e_recov = get_average_recov_elasticity()
a = a_average  # to be replaced by imported funciton value
S_Cu_max = 0.002002  # to be replaced by computed value from model
Q_1_3 = 700  # to be replaced by computed value from model
Q_2_3 = 300  # to be replaced by computed value from model
Q_3_14 = 2  # to be replaced by computed value from model
Q_6_14 = 3  # to be replaced by computed value from model
Q_EoL = 400  # to be replaced by computed value from model
S_reuse = cfg.s_reuse
S_EoL_net_T = 0  # to be replaced by computed value from model
exog_EAF = exog_EAF_price
P_PrSt = P_PrSt_price
Q_EoL_g = np.array([100, 50, 75, 175])  # to be replaced by values from model
S_Cu_alloy_g = np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p])  # config or model
S_Cu_0_g = np.array([0.003, 0.0025, 0.001, 0.002])  # cfg or model
R_0_recov_g = np.array(
    [cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction, cfg.r_0_recov_products])


# Functions
def R_recov_g(g_val, P_col):
    return 1 - (1 - R_0_recov_g[g_val]) * ((P_col / P_0_col + a) / (1 + a)) ** e_recov


c2 = (0.8 * S_Cu_max * (Q_1_3 + Q_2_3 - Q_3_14 - Q_6_14)) / (Q_EoL * (1 - S_reuse) * (1 + S_EoL_net_T))


def P_col_expr(S_Cu):
    c1 = ((np.sum(Q_EoL_g) / np.sum((1 - R_0_recov_g) * Q_EoL_g)) ** (1 / e_recov)) * (1 + a)
    test1 = (1 - c2 / S_Cu)
    test2 = (1 / e_recov)
    test3 = ((1 - c2 / S_Cu) ** (1 / e_recov))
    test4 = ((1 - c2 / S_Cu) ** (1 / e_recov) * c1 - a)

    P_col = ((1 - c2 / S_Cu) ** (1 / e_recov) * c1 - a) * P_0_col
    return P_col, c1


def P_dis_expr(S_Cu, P_col):
    numerator_P_dis_expr = S_Cu * np.sum([R_recov_g(i, P_col) * Q_EoL_g[i] for i in range(4)]) - np.sum(
        [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(4)])
    denominator_P_dis_expr = (1 / P_0_dis) ** e_dis * np.sum(
        [R_recov_g(i, P_col) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(4)])
    P_dis = (numerator_P_dis_expr / denominator_P_dis_expr) ** (1 / e_dis)
    return P_dis, numerator_P_dis_expr, denominator_P_dis_expr


def function(S_Cu):
    P_col, _ = P_col_expr(S_Cu)
    P_dis, _, _ = P_dis_expr(S_Cu, P_col)
    return P_col + P_dis + exog_EAF - P_PrSt


def derivative(S_Cu, epsilon=1e-6):
    return (function(S_Cu + epsilon) - function(S_Cu - epsilon)) / (2 * epsilon)


# Root finding methods
def find_root_bisection(a, b, tol=1e-6):
    return bisect(function, a, b, xtol=tol)


def find_root_newton(initial_guess, tol=1e-6):
    return newton(function, initial_guess, fprime=derivative, tol=tol)


# Usage
# start_value = c2*1.0000001
S_Cu_bisection = find_root_bisection(c2, 0.5)
S_Cu_newton = find_root_newton(S_Cu_bisection)

print('\n')
print(f"Root found by bisection method: {S_Cu_bisection}")
print(f"Root found by Newton-Raphson method: {S_Cu_newton}")

P_col, _ = P_col_expr(S_Cu_newton)
P_dis, _, _ = P_dis_expr(S_Cu_newton, P_col)

R_recov_1 = R_recov_g(0, P_col)
R_recov_2 = R_recov_g(1, P_col)
R_recov_3 = R_recov_g(2, P_col)
R_recov_4 = R_recov_g(3, P_col)

P_dis_col_eaf = P_col + P_dis + exog_EAF_price
price_diff = P_PrSt_price - P_dis_col_eaf

print(f"P_col: {P_col}")
print(f"P_dis: {P_dis}")
print(f"P_col + P_dis + exog(EAF) : {P_dis_col_eaf}")
print(f"P_Se_St - P_PrSt: {price_diff}")
print(f"R_recov transport: {R_recov_1}")
print(f"R_recov machinery: {R_recov_2}")
print(f"R_recov construction: {R_recov_3}")
print(f"R_recov products: {R_recov_4}")
