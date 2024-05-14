import os
import csv
import pandas as pd
import numpy as np
from src.tools.config import cfg

DAEHN_PATH = os.path.join(cfg.data_path, 'original', 'daehn')
CULLEN_PATH = os.path.join(cfg.data_path, 'original', 'cullen')


def get_cullen_fabrication_yield():
    # TODO make like forming yield, check usages, make np.array?
    fabrication_yield_path = os.path.join(CULLEN_PATH, 'cullen_fabrication_yield_matrix.csv')
    with open(fabrication_yield_path) as csv_file:
        cullen_reader = csv.reader(csv_file, delimiter=',')
        cullen_list = list(cullen_reader)
        fabrication_yield = [float(line[1]) for line in cullen_list[1:]]
    return np.array(fabrication_yield)


def get_cullen_forming_yield():
    forming_yield_path = os.path.join(CULLEN_PATH, 'cullen_forming_yield.csv')
    forming_yield = _load_normal_param_csv(forming_yield_path)
    return forming_yield


def get_cullen_production_yield():
    production_yield_path = os.path.join(CULLEN_PATH, 'cullen_production_yield.csv')
    production_yield = _load_normal_param_csv(production_yield_path)
    return production_yield


def get_daehn_tolerances():
    tolerances_path = os.path.join(DAEHN_PATH, 'cu_tolerances_ip.csv')
    tolerances = _load_normal_param_csv(tolerances_path)
    return tolerances


def get_daehn_intermediate_good_distribution():
    ig_distribution_path = os.path.join(DAEHN_PATH, 'intermediate_to_good_distribution.csv')
    ig_distribution = _load_normal_param_csv(ig_distribution_path)
    return ig_distribution


def get_daehn_good_intermediate_distribution():
    gi_distribution_path = os.path.join(DAEHN_PATH, 'good_to_intermediate_distribution.csv')
    gi_distribution = _load_normal_param_csv(gi_distribution_path)
    gi_distribution = gi_distribution.transpose()  # transpose matrix to have goods as first matrix
    return gi_distribution


def get_wittig_distributions():
    wittig_path = os.path.join(cfg.data_path, 'original', 'wittig', 'Wittig_matrix.csv')
    with open(wittig_path) as csv_file:
        wittig_reader = csv.reader(csv_file, delimiter=',')
        wittig_list = list(wittig_reader)
        use_recycling_params = [[float(num) for num in line[1:-1]] for line in wittig_list[1:]]
        recycling_usable_params = [float(line[-1]) for line in wittig_list[1:]]

    return use_recycling_params, recycling_usable_params


def get_worldsteel_intermediate_trade_shares():
    intermediate_trade_share_path = os.path.join(cfg.data_path, 'original', 'worldsteel', 'WS_digitalized',
                                                 'global_intermediate_trade_shares_cropped.csv')
    # we use the cropped trade shares so that the trade fits perfectly with the good - intermediate product distribution

    intermediate_trade_shares = _load_normal_param_csv(intermediate_trade_share_path)
    return intermediate_trade_shares


def _load_normal_param_csv(path):
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df = df.sort_index()
    df = df.sort_index(axis=1)
    data = df.to_numpy()
    if data.shape[-1] == 1:  # if shape of dimension of np-array is 1, dimension is not needed
        data = data.reshape(data.shape[:-1])
    return data


def _test():
    tolerances = get_daehn_tolerances()
    ig_distribution = get_daehn_intermediate_good_distribution()
    gi_distrubtion = get_daehn_good_intermediate_distribution()
    use_recycling_params, recycling_usable_params = get_wittig_distributions()
    fabrication_yield = get_cullen_fabrication_yield()
    forming_yield = get_cullen_forming_yield()
    production_yield = get_cullen_production_yield()

    print(f'Tolerances: \n{tolerances}\n')
    print(f'Intermediate-In_Use_Good-Distribution: \n{ig_distribution}\n')
    print(f'In_Use_Good-Intermediate-Distribution: \n{gi_distrubtion}\n')
    print(f'Use-Recycling-Distribution: \n{use_recycling_params}\n')
    print(f'Recycling-Usable-Distribution: \n{recycling_usable_params}\n')
    print(f'Fabrication_Yield: \n{fabrication_yield}\n')
    print(f'Forming Yield: \n{forming_yield}\n')
    print(f'Production Yield: \n{production_yield}\n')


if __name__ == '__main__':
    _test()
