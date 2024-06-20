import os
from src.tools.config import cfg
import pandas as pd
import numpy as np


def get_remind_prices():
    if cfg.price_scenario == '1_5_degree':
        p_steel = load_steel_prices_1p5_interpolated()
    elif cfg.price_scenario == 'baseline':
        p_steel = load_steel_prices_baseline_interpolated()
    else:
        raise RuntimeError(f'Price scenario is wrongly declared, hast to be baseline or 1_5_degree.\n'
                           f'Currently it is: {cfg.price_scenario}')
    p_steel = p_steel[-78:]  # only choose prices from 2023 on
    return p_steel


def get_remind_baseline_prices():
    return load_steel_prices_baseline_interpolated()[-78:]


def load_steel_prices_1p5_interpolated():
    file_path_1p5 = os.path.join(cfg.data_path, 'original', 'remind', 'steel_prices_1p5_interpolated.csv')
    df = pd.read_csv(file_path_1p5)
    prices_1p5 = np.zeros((81, 12))
    values = df['Value'].values
    rows_per_region = 81

    for i in range(12):
        start_row = i * rows_per_region
        end_row = start_row + rows_per_region
        prices_1p5[:, i] = values[start_row:end_row]

    return prices_1p5


def load_steel_prices_baseline_interpolated():
    file_path_baseline = os.path.join(cfg.data_path, 'original', 'remind', 'steel_prices_baseline_interpolated.csv')
    df = pd.read_csv(file_path_baseline)
    prices_baseline = np.zeros((81, 12))
    values = df['Value'].values
    rows_per_region = 81

    for i in range(12):
        start_row = i * rows_per_region
        end_row = start_row + rows_per_region
        prices_baseline[:, i] = values[start_row:end_row]

    return prices_baseline


def _test():
    prices_1p5 = load_steel_prices_1p5_interpolated()
    prices_baseline = load_steel_prices_baseline_interpolated()

    print("Shape of prices array baseline scenario:", load_steel_prices_1p5_interpolated().shape)
    print("First column of prices array 1p5 scenario:", load_steel_prices_1p5_interpolated()[:, 0])

    print("Shape of prices array baseline scenario:", load_steel_prices_baseline_interpolated().shape)
    print("First column of prices array baseline scenario:", load_steel_prices_baseline_interpolated()[:, 0])


if __name__ == '__main__':
    _test()
