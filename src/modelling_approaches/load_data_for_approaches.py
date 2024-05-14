import numpy as np
from src.tools.config import cfg
from src.calc_trade.calc_indirect_trade import get_scaled_past_indirect_trade
from src.calc_trade.calc_trade import get_scaled_past_trade
from src.tools.tools import get_np_from_df
from src.read_data.load_data import load_production
from src.read_data.load_data import load_stocks


def get_past_production_trade_forming_fabrication(country_specific):
    production = get_past_production(country_specific)
    trade = get_scaled_past_trade(country_specific, scaler=production, intermediate_resolution=True)
    indirect_trade = get_scaled_past_indirect_trade(country_specific, scaler=production)

    return production, trade, indirect_trade
    # TODO check usages, deleted forming fabrication here because one can't know with intermmediate products


def get_past_production(country_specific):
    df_production = load_production(country_specific=country_specific).transpose()
    production = get_np_from_df(df_production, data_split_into_categories=False)
    # production = production[:109]  # only use data up to 2008 TODO: delete/decide
    return production


def get_past_stocks(country_specific):  # TODO put all data load function into different file?
    df_stocks = load_stocks(country_specific=country_specific, per_capita=False)
    stocks = get_np_from_df(df_stocks, data_split_into_categories=True)
    stocks = np.moveaxis(stocks, -1, 0)  # move time axis to first position for 'trg' format

    return stocks
