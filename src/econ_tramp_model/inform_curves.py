import sys
import math
import numpy as np
from matplotlib import pyplot as plt

'''# Add the specific directory to the Python path
script_directory = '/Users/marcelgeller/PycharmProjects/curve_informing_simson/venv/Testings/curve_informing'
sys.path.append(script_directory)'''
from config import cfg

# Define parameter inputs for recovery price elasticity for all sectors
# Transport
r0_T = cfg.r_0_recov_transport
rp_T = cfg.rp_transport
p0_T = cfg.p0_transport
pp_T = cfg.pp_transport
r_free_t = cfg.r_free_transport
e_recov_T = math.log((-rp_T + 1) / (-r0_T + 1)) / math.log(pp_T / p0_T)

# Machinery
r0_M = cfg.r_0_recov_machinery
rp_M = cfg.rp_machinery
p0_M = cfg.p0_machinery
pp_M = cfg.pp_machinery
r_free_m = cfg.r_free_machinery
e_recov_M = math.log((-rp_M + 1) / (-r0_M + 1)) / math.log(pp_M / p0_M)

# Construction
r0_C = cfg.r_0_recov_construction
rp_C = cfg.rp_construction
p0_C = cfg.p0_construction
pp_C = cfg.pp_construction
r_free_c = cfg.r_free_construction
e_recov_C = math.log((-rp_C + 1) / (-r0_C + 1)) / math.log(pp_C / p0_C)

# Products
r0_P = cfg.r_0_recov_products
rp_P = cfg.rp_products
p0_P = cfg.p0_products
pp_P = cfg.pp_products
r_free_p = cfg.r_free_products
e_recov_P = math.log((-rp_P + 1) / (-r0_P + 1)) / math.log(pp_P / p0_P)


# Calculate average price elasticity of scrap recovery from all sectors
def get_average_recov_elasticity():
    e_recov = np.average([e_recov_T, e_recov_M, e_recov_C, e_recov_P])
    return e_recov


e_recov = get_average_recov_elasticity()


def get_parameters_a_for_recov_curves():
    a_t = 1 / (((1 - r0_T) / (1 - r_free_t)) ** (1 / e_recov) - 1)
    a_m = 1 / (((1 - r0_M) / (1 - r_free_m)) ** (1 / e_recov) - 1)
    a_c = 1 / (((1 - r0_C) / (1 - r_free_c)) ** (1 / e_recov) - 1)
    a_p = 1 / (((1 - r0_P) / (1 - r_free_p)) ** (1 / e_recov) - 1)
    a_g_values = [a_t, a_m, a_c, a_p]
    return a_g_values


a_average = np.sum(get_parameters_a_for_recov_curves()) / 4
print('test:', a_average)

a_g_values = get_parameters_a_for_recov_curves()
# print(get_parameters_a_for_recov_curves())
# print(a_g_values)

### Define parameter inputs for copper share price elasticity for all sectors
# Transport
S_Cu_0_T = cfg.s_cu_0_transport
S_Cu_p_T = cfg.s_cu_p_transport
e_dis_T = math.log(S_Cu_0_T / S_Cu_p_T) / math.log(p0_T / pp_T)

# Machinery
S_Cu_0_M = cfg.s_cu_0_machinery
S_Cu_p_M = cfg.s_cu_p_machinery
e_dis_M = math.log(S_Cu_0_M / S_Cu_p_M) / math.log(p0_M / pp_M)

# Construction
S_Cu_0_C = cfg.s_cu_0_construction
S_Cu_p_C = cfg.s_cu_p_construction
e_dis_C = math.log(S_Cu_0_C / S_Cu_p_C) / math.log(p0_C / pp_C)

# Products
S_Cu_0_P = cfg.s_cu_0_products
S_Cu_p_P = cfg.s_cu_p_products
e_dis_P = math.log(S_Cu_0_P / S_Cu_p_P) / math.log(p0_P / pp_P)


# Calculate average copper share price elasticity from all sectors
def get_average_dis_elasticity():
    e_dis = np.average([e_dis_T, e_dis_M, e_dis_C, e_dis_P])
    return e_dis


e_dis = get_average_dis_elasticity()


def main():
    print('1. RECOVERY RATE\n')
    print('The average price elasticity of scrap recovery across all sectors (T, M, C, P) is: ', e_recov, '\n')

    ### RECOVERY RATE TRANSPORT AND MACHINERY
    print('\033[4mRECOVERY RATE TRANSPORT & MACHINERY\033[0m\n')

    r0 = r0_T
    rp = rp_T
    p0 = p0_T
    pp = pp_T
    r_free = 0.

    a_t = 1 / (((1 - r0) / (1 - r_free)) ** (1 / e_recov) - 1)
    print('Value for parameter a: ', a_t)

    def r(p):
        return (1 - (1 - r0) * ((p / p0 + a_average) / (1 + a_average)) ** e_recov)

    print('Value for base recovery rate: ', r0)
    print('Value for recovery rate when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0., 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('Scrap Recovery Rate Transport & Machinery')
    plt.xlabel('Recovery Rate [%]')
    plt.ylabel(r'Collection Costs P_col [$/t]')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 1.])
    plt.ylim([0, 1000])
    plt.text(0.05, 850, f'Elasticity T,M: {e_recov:.2f}', fontsize=9, color='black')
    plt.text(0.05, 800, f'Value a: {a_average:.2f}', fontsize=9, color='black')
    plt.text(0.05, 750, f'R_recov base: {r0:.2f}', fontsize=9, color='black')
    plt.text(0.05, 700, f'R_recov if P_col doubles: {r(300):.2f}', fontsize=9, color='black')
    plt.show()

    ## RECOVERY RATE CONSTRUCTION
    print('\033[4mRECOVERY RATE CONSTRUCTION\033[0m\n')
    r0 = r0_C
    rp = rp_C
    p0 = p0_C
    pp = pp_C
    r_free = 0.

    a_c = 1 / (((1 - r0) / (1 - r_free)) ** (1 / e_recov) - 1)
    print('Value for parameter a: ', a_average)

    def r(p):
        return (1 - (1 - r0) * ((p / p0 + a_average) / (1 + a_average)) ** e_recov)

    print('Value for base recovery rate: ', r0)
    print('Value for recovery rate when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0., 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('Scrap Recovery Rate Construction')
    plt.xlabel('Recovery Rate [%]')
    plt.ylabel(r'Collection Costs P_col [$/t]')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 1.])
    plt.ylim([0, 1000])
    plt.text(0.05, 850, f'Elasticity C: {e_recov:.2f}', fontsize=9, color='black')
    plt.text(0.05, 800, f'Value a: {a_average:.2f}', fontsize=9, color='black')
    plt.text(0.05, 750, f'R_recov base: {r0:.2f}', fontsize=9, color='black')
    plt.text(0.05, 700, f'R_recov if P_col doubles: {r(300):.2f}', fontsize=9, color='black')
    plt.show()

    ## RECOVERY RATE PRODUCTS
    print('\033[4mRECOVERY RATE PRODUCTS\033[0m\n')

    r0 = r0_P
    rp = rp_P
    p0 = p0_P
    pp = pp_P
    r_free = 0.

    a_p = 1 / (((1 - r0) / (1 - r_free)) ** (1 / e_recov) - 1)
    print('Value for parameter a: ', a_p)

    def r(p):
        return (1 - (1 - r0) * ((p / p0 + a_average) / (1 + a_average)) ** e_recov)

    print('Value for base recovery rate: ', r0)
    print('Value for recovery rate when price doubles: ', r(300), '\n\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0., 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('Scrap Recovery Rate Products')
    plt.xlabel('Recovery Rate [%]')
    plt.ylabel(r'Collection Costs P_col [$/t]')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 1.])
    plt.ylim([0., 1000])
    plt.text(0.05, 850, f'Elasticity P: {e_recov:.2f}', fontsize=9, color='black')
    plt.text(0.05, 800, f'Value a: {a_average:.2f}', fontsize=9, color='black')
    plt.text(0.05, 750, f'R_recov base: {r0:.2f}', fontsize=9, color='black')
    plt.text(0.05, 700, f'R_recov if P_col doubles: {r(300):.2f}', fontsize=9, color='black')
    plt.show()

    print('---------------------------------------------------------------------------------------------\n\n')
    print('2. EXTERNAL COPPER SHARE ADDED IN RECYCLING\n')
    print(
        'The average price elasticity of external copper share added in recycling across all sectors (T, M, C, P) is: ',
        e_dis, '\n')

    # External Copper Share added in Recycling Transport
    print('\033[4mEXTERNAL COPPER SHARE ADDED IN RECYCLING (TRANSPORT)\033[0m\n')
    S_Cu_0_T_adjusted = S_Cu_0_T / p0_T ** e_dis

    def r(p):
        return S_Cu_0_T_adjusted * p ** e_dis

    print('Value for base copper share: ', S_Cu_0_T)
    print('Value for base copper share adjusted due to average price elasticity: ', S_Cu_0_T_adjusted)
    print('Value for copper share when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0.000001, 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('External Copper Share added in Recycling (Transport)')
    plt.xlabel('Copper Share S_Cu,T added [wt%]')
    plt.ylabel(r'Disassembly Costs $/t')
    plt.axvline(x=0.3, color='black', linestyle='--')
    plt.text(0.31, 400, 'copper limit\ntransport')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 0.5])
    plt.ylim([0., 1000])
    plt.text(0.05, 850, f'Elasticity: {e_dis:.2f}', fontsize=9, color='black')
    plt.show()

    ### External Copper Share added in Recycling MACHINERY
    print('\033[4mEXTERNAL COPPER SHARE ADDED IN RECYCLING (MACHINERY)\033[0m\n')
    S_Cu_0_M_adjusted = S_Cu_0_M / p0_M ** e_dis

    def r(p):
        return S_Cu_0_M_adjusted * p ** e_dis

    print('Value for base copper share: ', S_Cu_0_M)
    print('Value for base copper share adjusted due to average price elasticity: ', S_Cu_0_M_adjusted)
    print('Value for copper share when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0.000001, 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('External Copper Share added in Recycling (Machinery)')
    plt.xlabel('Copper Share S_Cu,M added [wt%]')
    plt.ylabel(r'Disassembly Costs $/t')
    plt.axvline(x=0.25, color='black', linestyle='--')
    plt.text(0.26, 300, 'copper limit\nmachinery')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 0.5])
    plt.ylim([0., 1000])
    plt.text(0.05, 850, f'Elasticity: {e_dis:.2f}', fontsize=9, color='black')
    plt.show()

    ### External Copper Share added in Recycling CONSTRUCTION
    print('\033[4mEXTERNAL COPPER SHARE ADDED IN RECYCLING (CONSTRUCTION)\033[0m\n')
    S_Cu_0_C_adjusted = S_Cu_0_C / p0_C ** e_dis

    def r(p):
        return 2337.13 * p ** e_dis

    print('Value for base copper share: ', S_Cu_0_C)
    print('Value for base copper share adjusted due to average price elasticity: ', S_Cu_0_C_adjusted)
    print('Value for copper share when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0.000001, 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('External Copper Share added in Recycling (Construction)')
    plt.xlabel('Copper Share S_Cu,C added [wt%]')
    plt.ylabel(r'Disassembly Costs $/t')
    plt.axvline(x=0.1, color='black', linestyle='--')
    plt.text(0.11, 300, 'copper limit\nconstruction')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 0.5])
    plt.ylim([0., 1000])
    plt.text(0.4, 850, f'Elasticity: {e_dis:.2f}', fontsize=9, color='black')
    plt.show()

    ### External Copper Share added in Recycling PRODUCTS
    print('\033[4mEXTERNAL COPPER SHARE ADDED IN RECYCLING (PRODUCTS)\033[0m\n')
    S_Cu_0_P_adjusted = S_Cu_0_P / p0_P ** e_dis

    def r(p):
        return S_Cu_0_P_adjusted * p ** e_dis

    print('Value for base copper share: ', S_Cu_0_P)
    print('Value for base copper share adjusted due to average price elasticity: ', S_Cu_0_P_adjusted)
    print('Value for copper share when price doubles: ', r(300), '\n')

    plt.figure(figsize=(7, 6))
    p = np.arange(0.000001, 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('External Copper Share added in Recycling (Products)')
    plt.xlabel('Copper Share S_Cu,P added [wt%]')
    plt.ylabel(r'Disassembly Costs $/t')
    plt.axvline(x=0.4, color='black', linestyle='--')
    plt.text(0.275, 300, 'copper limit\nproducts')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 0.5])
    plt.ylim([0., 1000])
    plt.text(0.05, 450, f'Elasticity: {e_dis:.2f}', fontsize=9, color='black')
    plt.show()

    ### Price Sensitivity of Steel Demand
    ela = -0.3

    def r(p):
        return p ** ela

    plt.figure(figsize=(7, 6))
    p = np.arange(0.000001, 1000, 0.01)
    plt.plot(r(p), p)
    plt.title('Price Sensitivity of Steel Demand')
    plt.xlabel('Effect of Price on Demand [%]')
    plt.ylabel(r'Steel Price $/t')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0., 2])
    plt.ylim([0., 1000])
    plt.text(0.25, 850, f'Elasticity: {ela:.2f}', fontsize=9, color='black')
    plt.show()


if __name__ == "__main__":
    main()
