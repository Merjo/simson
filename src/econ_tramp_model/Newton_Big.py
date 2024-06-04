import sys
import os
import sympy as sp

# Add the specific directory to the Python path
script_directory1 = '/Users/marcelgeller/PycharmProjects/curve_informing_simson/venv/Testings/curve_informing'
sys.path.append(script_directory1)
from config import cfg

# Import e_recov from plot_all_curves_informed.py
from inform_curves import get_average_recov_elasticity, get_average_dis_elasticity, get_parameters_a_for_recov_curves, \
    a_average
from steel_price_curves import get_price_for_scenario_and_year

# defined year
t_price = 2050

# Call the price function to get prices
exog_EAF_price, P_PrSt_price = get_price_for_scenario_and_year(t_price)
print('price of eaf: ', exog_EAF_price)
print('price of bof: ', P_PrSt_price)

# Define the constants and variables as provided
P_0_col = sp.symbols('P_0_col', constant=True)
P_0_dis = sp.symbols('P_0_dis', constant=True)
e_dis = sp.symbols('e_dis', constant=True)
e_recov = sp.symbols('e_recov', constant=True)
a = sp.symbols('a', constant=True)
S_Cu_max = sp.symbols('S_Cu_max', constant=True)
Q_1_3, Q_2_3, Q_3_14, Q_6_14, Q_EoL = sp.symbols('Q_1_3 Q_2_3 Q_3_14 Q_6_14 Q_EoL', constant=True)
S_reuse, S_EoL_net_T = sp.symbols('S_reuse S_EoL_net_T', constant=True)
exog_EAF, P_PrSt = sp.symbols('exog_EAF P_PrSt', constant=True)
g = sp.symbols('g', integer=True)
S_Cu = sp.symbols('S_Cu')

# Define indexed symbols for Q_EoL_g, S_Cu_alloy_g, S_Cu_0_g, and R_0_recov_g
Q_EoL_g = sp.IndexedBase('Q_EoL_g')
S_Cu_alloy_g = sp.IndexedBase('S_Cu_alloy_g')
S_Cu_0_g = sp.IndexedBase('S_Cu_0_g')
R_0_recov_g = sp.IndexedBase('R_0_recov_g')
Q_EoL_values = [100, 50, 75, 175]  # to be replaced by values from model
S_Cu_alloy_values = [cfg.s_cu_alloy_t, cfg.s_cu_alloy_m, cfg.s_cu_alloy_c, cfg.s_cu_alloy_p]  # config or model
S_Cu_0_values = [0.003, 0.003, 0.003, 0.003]
R_0_recov_values = [cfg.r_0_recov_transport, cfg.r_0_recov_machinery, cfg.r_0_recov_construction,
                    cfg.r_0_recov_products]

# Define the intermediate expressions and functions as provided
sum_Q_EoL_g = sum([Q_EoL_g[i] for i in range(0, 4)])
sum_R_0_Q_EoL_g = sum([(1 - R_0_recov_g[i]) * Q_EoL_g[i] for i in range(0, 4)])

c1 = ((sum_Q_EoL_g / sum_R_0_Q_EoL_g) ** (1 / e_recov)) * (1 + a)
c2 = (0.8 * S_Cu_max * (Q_1_3 + Q_2_3 - Q_3_14 - Q_6_14)) / (Q_EoL * (1 - S_reuse) * (1 + S_EoL_net_T))
P_col = ((1 - c2 / S_Cu) ** (1 / e_recov) * c1 - a) * P_0_col


# Adjusted R_recov_g to be indexed
def R_recov_g_expr(g_val):
    return 1 - (1 - R_0_recov_g[g_val]) * ((P_col / P_0_col + a) / (1 + a)) ** e_recov


numerator_P_dis_expr = S_Cu * sum([R_recov_g_expr(i) * Q_EoL_g[i] for i in range(0, 4)]) - sum(
    [R_recov_g_expr(i) * Q_EoL_g[i] * S_Cu_alloy_g[i] for i in range(0, 4)])
denominator_P_dis_expr = (1 / P_0_dis) ** e_dis * sum(
    [R_recov_g_expr(i) * Q_EoL_g[i] * S_Cu_0_g[i] for i in range(0, 4)])
P_dis_expr = (numerator_P_dis_expr / denominator_P_dis_expr) ** (1 / e_dis)

function = P_col + P_dis_expr + exog_EAF - P_PrSt
function_prime = sp.diff(function, S_Cu)


def bisect(f, a0, b, tol=1e-5, max_iter=100):
    fa0 = f.evalf(subs={S_Cu: a0})
    fb = f.evalf(subs={S_Cu: b})

    if fa0.as_real_imag()[1] != 0 or fb.as_real_imag()[1] != 0:
        raise ValueError(f"Function evaluates to complex values at the interval endpoints. fa0: {fa0}, fb: {fb}")

    fa0 = fa0.as_real_imag()[0]
    fb = fb.as_real_imag()[0]

    if fa0 * fb > 0:
        raise ValueError("The function has the same sign at the end points of the interval [a, b].")

    for _ in range(max_iter):
        c = (a0 + b) / 2
        fc = f.evalf(subs={S_Cu: c})

        if fc.as_real_imag()[1] != 0:
            raise ValueError(f"Function evaluates to a complex value within the interval at c={c}. fc: {fc}")

        fc = fc.as_real_imag()[0]

        if abs(fc) < tol or abs(b - a0) / 2 < tol:
            return c

        if fa0 * fc < 0:
            b = c
        else:
            a0 = c
            fa0 = fc

    return (a0 + b) / 2


def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    x = x0
    for i in range(max_iter):
        f_val = f.evalf(subs={S_Cu: x})
        df_val = df.evalf(subs={S_Cu: x})

        if f_val.as_real_imag()[1] != 0 or df_val.as_real_imag()[1] != 0:
            raise ValueError(
                f"Function or its derivative evaluates to a complex value. f_val: {f_val}, df_val: {df_val}")

        f_val = f_val.as_real_imag()[0]
        df_val = df_val.as_real_imag()[0]

        print(f"Iteration {i + 1}: S_Cu = {x}, f(S_Cu) = {f_val}, f'(S_Cu) = {df_val}")

        if df_val == 0:
            raise ValueError("Derivative is zero. No solution found.")

        x_new = x - f_val / df_val

        if not 0 < x_new < 1:
            raise ValueError(f"S_Cu out of the range [0, 1]. Current value: {x_new}")

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    raise ValueError("Maximum number of iterations exceeded. No solution found.")


constants = {
    P_0_col: cfg.p_0_col,
    P_0_dis: cfg.p_0_dis,
    e_dis: get_average_dis_elasticity(),
    e_recov: get_average_recov_elasticity(),
    a: a_average,  # to be replaced by imported funciton value
    S_Cu_max: 0.002002,  # to be replaced by computed value from model
    Q_1_3: 700,  # to be replaced by computed value from model
    Q_2_3: 300,  # to be replaced by computed value from model
    Q_3_14: 2,  # to be replaced by computed value from model
    Q_6_14: 3,  # to be replaced by computed value from model
    Q_EoL: 400,  # to be replaced by computed value from model
    S_reuse: cfg.s_reuse,
    S_EoL_net_T: 0,  # to be replaced by computed value from model
    exog_EAF: exog_EAF_price,
    P_PrSt: P_PrSt_price
}

# Substitute values for Q_EoL_g, S_Cu_alloy_g, S_Cu_0_g, and R_0_recov_g
constants.update({Q_EoL_g[i]: Q_EoL_values[i] for i in range(0, 4)})
constants.update({S_Cu_alloy_g[i]: S_Cu_alloy_values[i] for i in range(0, 4)})
constants.update({S_Cu_0_g[i]: S_Cu_0_values[i] for i in range(0, 4)})
constants.update({R_0_recov_g[i]: R_0_recov_values[i] for i in range(0, 4)})

# S_Cu_0_values = [0.003, 0.0025, 0.001, 0.004]

# Substitute sums with explicit values
f = function.subs(constants).subs({sum_Q_EoL_g: sum(Q_EoL_values),
                                   sum_R_0_Q_EoL_g: sum(
                                       (1 - constants[R_0_recov_g[i]]) * v for i, v in enumerate(Q_EoL_values))})
df = function_prime.subs(constants).subs({sum_Q_EoL_g: sum(Q_EoL_values),
                                          sum_R_0_Q_EoL_g: sum(
                                              (1 - constants[R_0_recov_g[i]]) * v for i, v in enumerate(Q_EoL_values))})

a0 = c2.subs(constants)
b = 0.5

try:
    x0 = bisect(f, a0, b)
    root = newton_raphson(f, df, x0)
    print(f"The root is: {root}")

    # Calculate P_col, P_dis, and R_recov_g using the root found
    P_col_val = P_col.subs({**constants, S_Cu: root, sum_Q_EoL_g: sum(Q_EoL_values),
                            sum_R_0_Q_EoL_g: sum(
                                (1 - constants[R_0_recov_g[i]]) * v for i, v in enumerate(Q_EoL_values))})

    # Ensure R_0_recov_g is substituted correctly
    R_recov_g_vals = [R_recov_g_expr(i).subs(
        {**constants, S_Cu: root, P_col: P_col_val, R_0_recov_g[i]: constants[R_0_recov_g[i]]}).evalf() for i in
                      range(0, 4)]

    numerator_P_dis_val = numerator_P_dis_expr.subs({**constants, S_Cu: root, sum_Q_EoL_g: sum(Q_EoL_values),
                                                     sum_R_0_Q_EoL_g: sum(
                                                         (1 - constants[R_0_recov_g[i]]) * v for i, v in
                                                         enumerate(Q_EoL_values)),
                                                     sp.Sum(R_recov_g_expr(g) * Q_EoL_g[g], (g, 0, 4)): sum(
                                                         R_recov_g_val * v for R_recov_g_val, v in
                                                         zip(R_recov_g_vals, Q_EoL_values)),
                                                     sp.Sum(R_recov_g_expr(g) * Q_EoL_g[g] * S_Cu_alloy_g[g],
                                                            (g, 0, 4)): sum(
                                                         R_recov_g_val * v * w for R_recov_g_val, v, w in
                                                         zip(R_recov_g_vals, Q_EoL_values, S_Cu_alloy_values))})

    denominator_P_dis_val = denominator_P_dis_expr.subs({**constants, S_Cu: root, sum_Q_EoL_g: sum(Q_EoL_values),
                                                         sum_R_0_Q_EoL_g: sum(
                                                             (1 - constants[R_0_recov_g[i]]) * v for i, v in
                                                             enumerate(Q_EoL_values)),
                                                         sp.Sum(R_recov_g_expr(g) * Q_EoL_g[g] * S_Cu_0_g[g],
                                                                (g, 0, 4)): sum(
                                                             R_recov_g_val * v * w for R_recov_g_val, v, w in
                                                             zip(R_recov_g_vals, Q_EoL_values, S_Cu_0_values))})

    P_dis_val = (numerator_P_dis_val / denominator_P_dis_val) ** (1 / constants[e_dis])

    print(f"P_col: {P_col_val}")
    print(f"P_dis: {P_dis_val}")
    print(f"R_recov_g: {R_recov_g_vals}")

    # Check the sum P_col + P_dis + exog_EAF
    sum_check = P_col_val + P_dis_val + constants[exog_EAF]
    print(f"Sum (P_col + P_dis + exog_EAF): {sum_check}")
    print(f"P_PrSt: {constants[P_PrSt]}")
    print(f"Price Difference P_PrSt and _SeSt: {sum_check - constants[P_PrSt]}")

except ValueError as e:
    print(f"Error during root finding: {e}")
