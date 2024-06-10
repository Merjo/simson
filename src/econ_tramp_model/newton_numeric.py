import numpy as np
from scipy.optimize import bisect, newton
import sys
import os
import matplotlib.pyplot as plt
import sympy as sp

# Add the specific directory to the Python path
script_directory1 = '/Users/marcelgeller/PycharmProjects/curve_informing_simson/venv/Testings/curve_informing'
sys.path.append(script_directory1)
from config import cfg

# Import important parameters and functions
from inform_curves import get_average_recov_elasticity, get_average_dis_elasticity, get_parameters_a_for_recov_curves, \
    a_average
from steel_price_curves import get_price_for_scenario_and_year

print('e_dis ela: ', get_average_dis_elasticity())
print('e_recov ela: ', get_average_recov_elasticity())

# defined year
t_price = 2050

# Call the price function to get prices
exog_EAF_price, P_PrSt_price = get_price_for_scenario_and_year(t_price)

# Constants
P_0_col = cfg.p_0_col  # 150 #cfg.p_0_col
P_0_dis = cfg.p_0_dis  # 150 #cfg.p_0_dis
e_dis = get_average_dis_elasticity()  # -2.548650743789492 #get_average_dis_elasticity()
e_recov = get_average_recov_elasticity()  # -0.8304820237218398 #get_average_recov_elasticity()
a = a_average  # 0.25340259071206267 #a_average
S_Cu_max = 0.0002  # 0.0004 is the minimum value S_Cu_max can obtain as lowest tolerance is 0.04wt%
Q_1_3 = 700
Q_2_3 = 300
Q_3_14 = 2
Q_6_14 = 3
Q_EoL = 400
S_reuse = cfg.s_reuse  # cfg.s_reuse #0
S_EoL_net_T = 0
exog_EAF = exog_EAF_price  # 150 #exog_EAF_price
P_PrSt = P_PrSt_price  # 600 #P_PrSt_price
Q_EoL_g = np.array([100, 50, 75, 175])
S_Cu_alloy_g = np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c,
                         cfg.s_cu_alloy_p])  # np.array([0.8 * 0.002002, 0.8 * 0.002465341, 0.8 * 0.001822651, 0.8 * 0.001985311]) #np.array([cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p])
S_Cu_0_g = np.array([cfg.s_cu_0_transport, cfg.s_cu_0_machinery, cfg.s_cu_0_construction,
                     cfg.s_cu_0_products])  # np.array([0.003, 0.0025, 0.001, 0.004]) #np.array([cfg.s_cu_0_transport, cfg.s_cu_0_machinery, cfg.s_cu_0_construction, cfg.s_cu_0_products])
R_0_recov_g = np.array([cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction,
                        cfg.r_0_recov_products])  # np.array([0.9, 0.9, 0.85, 0.5]) #np.array([cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction, cfg.r_0_recov_products])
S_Cu = sp.symbols('S_Cu')


# Functions
def R_recov_g(g_val, P_col):
    return 1 - (1 - R_0_recov_g[g_val]) * ((P_col / P_0_col + a) / (1 + a)) ** e_recov


# variables for P_col
c1 = ((np.sum(Q_EoL_g) / np.sum((1 - R_0_recov_g) * Q_EoL_g)) ** (1 / e_recov)) * (1 + a)

c2 = (0.8 * S_Cu_max * (Q_1_3 + Q_2_3 - Q_3_14 - Q_6_14)) / (Q_EoL * (1 - S_reuse) * (1 + S_EoL_net_T))
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


def find_newton_numerator(numerator_P_dis_expr, initial_guess):
    try:
        root_newton_numerator = newton(numerator_P_dis_expr, initial_guess) * 1.00001
        print(f"The root of the numerator of P_dis is: {root_newton_numerator}")
    except RuntimeError:
        print('No root exists for the numerator with the given parameters!')
        root_newton_numerator = 0
    return root_newton_numerator


root_newton_numerator = find_newton_numerator(numerator_P_dis_expr, initial_guess)

# Finding root of P_dis_ denominator Newton
root_newton_denominator = newton(denominator_P_dis_expr, initial_guess) * 0.99999
print(f"The root of the denominator of P_dis is: {root_newton_denominator}")


def decide_starting_values_for_bisec(root_newton_numerator, root_newton_denominator, c2_value_exceeded):
    if root_newton_numerator == 0:
        start_value_bisec = c2_value_exceeded
        print(
            f"The root of the numerator is zero, thus the new starting value for bisec is the value of c2*1.00001: {start_value_bisec}")
    elif root_newton_denominator > root_newton_numerator:
        start_value_bisec = root_newton_numerator
        print(
            f"The root of the numerator is smaller than the root of the denominator, thus the new starting value for bisec is: {start_value_bisec}")
    else:
        print("Error during root finding")
    return start_value_bisec


start_value_bisec = decide_starting_values_for_bisec(root_newton_numerator, root_newton_denominator, initial_guess)
# print('The new starting value for bisec is: ', start_value_bisec)

# Plots of P_dis, P_dis_numerator, P_dis_denominator
# Generate a range of S_Cu values for plotting numerator and denominator
S_Cu_values_numerator = np.linspace(initial_guess, 0.1, 500)
numerator_values = [numerator_P_dis_expr(S_Cu) for S_Cu in S_Cu_values_numerator]

# Generate a range of S_Cu values for plotting numerator and denominator
S_Cu_values_denominator = np.linspace(initial_guess, 0.1, 500)
denominator_values = [denominator_P_dis_expr(S_Cu) for S_Cu in S_Cu_values_denominator]

# Generate a range of S_Cu values for plotting P_dis
S_Cu_values_p_dis = np.linspace(0.000001, 0.1, 100)
P_dis_values = [P_dis_expr(S_Cu, P_col_expr(S_Cu_max)) for S_Cu in S_Cu_values_p_dis]

### Plot P_dis curves
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Plot numerator
ax1.plot(S_Cu_values_numerator, numerator_values, label='numerator_P_dis_expr')
ax1.set_xlabel('S_Cu')
ax1.set_ylabel('numerator_P_dis_expr')
ax1.set_title('numerator_P_dis_expr vs S_Cu')
ax1.legend()
ax1.grid(True)

# Plot denominator
ax2.plot(S_Cu_values_denominator, denominator_values, label='denominator_P_dis_expr')
ax2.set_xlabel('S_Cu')
ax2.set_ylabel('denominator_P_dis_expr')
ax2.set_title('denominator_P_dis_expr vs S_Cu')
ax2.legend()
ax2.grid(True)

# Plot P_dis
ax3.plot(S_Cu_values_p_dis, P_dis_values, label='P_dis')
ax3.set_xlabel('S_Cu')
ax3.set_ylabel('P_dis')
ax3.set_title('Plot of P_dis as a function of S_Cu')
# ax3.set_xlim(0, 1)
# ax3.set_ylim(0,100)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Plot R_recov_g functions
# Values for P_col
P_col_values = np.linspace(1, 1000, 100)

# Create subplots
fig, axs = plt.subplots(1, len(R_0_recov_g), figsize=(20, 5))

for i in range(len(R_0_recov_g)):
    R_recov_values = [R_recov_g(i, P_col) for P_col in P_col_values]
    axs[i].plot(R_recov_values, P_col_values)
    axs[i].set_title(f'R_recov_g for g_val={i}')
    axs[i].set_xlabel('R_recov_g')
    axs[i].set_ylabel('P_col')
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(0, 1000)
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# Usage
S_Cu_bisection = find_root_bisection(start_value_bisec, root_newton_denominator)
S_Cu_newton = find_root_newton(S_Cu_bisection)

print('\n')
print(f"Root found by bisection method: {S_Cu_bisection}")
print(f"Root found by Newton-Raphson method: {S_Cu_newton}")

P_col = P_col_expr(S_Cu_newton)
P_dis = P_dis_expr(S_Cu_newton, P_col)

R_recov_t = R_recov_g(0, P_col)
R_recov_m = R_recov_g(1, P_col)
R_recov_c = R_recov_g(2, P_col)
R_recov_p = R_recov_g(3, P_col)

P_dis_col_eaf = P_col + P_dis + exog_EAF
price_diff = P_PrSt - P_dis_col_eaf

print(f"P_col: {P_col}")
print(f"P_dis: {P_dis}")
print(f"P_col + P_dis + exog(EAF) : {P_dis_col_eaf}")
print(f"P_Se_St - P_PrSt: {price_diff}")
print(f"R_recov transport: {R_recov_t}")
print(f"R_recov machinery: {R_recov_m}")
print(f"R_recov construction: {R_recov_c}")
print(f"R_recov products: {R_recov_p}")
