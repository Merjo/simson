import numpy as np
import pandas as pd
from src.predict.prediction_tools import split_future_stocks_to_base_year_categories, \
    copy_stocks_across_scenarios
from src.tools.config import cfg
from darts.metrics import rmse
from darts.models import RNNModel
from darts import TimeSeries


def predict_lstm(stocks, gdp_data):
    print(stocks.shape)
    print(gdp_data.shape)
    past_stocks_by_category = stocks.copy()  # TODO dis dubble?
    stocks = np.sum(stocks, axis=2)

    gdp_data = gdp_data[:, :, 1]  # only use SSP2 TODO change
    gdp_data_future = gdp_data[109:]
    gdp_data_past = gdp_data[:109]  # up until 2009 all scenarios are the same

    future_stocks = _lstm_stock_curve(stocks, gdp_data)
    if cfg.include_gdp_and_pop_scenarios_in_prediction:
        future_stocks = np.moveaxis(future_stocks, 0, -1)

    future_stocks = split_future_stocks_to_base_year_categories(past_stocks_by_category, future_stocks,
                                                                is_future_stocks_with_scenarios=cfg.include_gdp_and_pop_scenarios_in_prediction)
    if cfg.include_gdp_and_pop_scenarios_in_prediction:
        past_stocks_by_category = copy_stocks_across_scenarios(past_stocks_by_category)

    stocks = np.concatenate([past_stocks_by_category, future_stocks], axis=0)

    return stocks


def _lstm_stock_curve(stocks, gdp):
    model = RNNModel(input_chunk_length=109,
                     output_chunk_length=92,
                     n_rnn_layers=2)
    stock_times = pd.date_range("1900-01-01", periods=109, freq="Y")
    gdp_times = pd.date_range("1900-01-01", periods=201, freq="Y")

    ts_stocks = TimeSeries.from_times_and_values(stock_times, stocks[:, 0])
    ts_gdp = TimeSeries.from_times_and_values(gdp_times, gdp[:, 0])
    model.fit(ts_stocks,
              future_covariates=ts_gdp,
              epochs=10,
              verbose=True)
    eval_model(model, ts_stocks)


def eval_model(model, stocks, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=stocks,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.54,  # should denote 2008 ((2008-1900)/(2100-1900)=0.54)
                                          retrain=False,
                                          verbose=True)

    stocks[-len(backtest) - 100:].plot()
    backtest.plot(label='backtest (n=10)')
    print('Backtest RMSE = {}'.format(rmse(stocks, backtest)))


if __name__ == '__main__':
    from src.predict.calc_steel_stocks import test

    test(strategy='LSTM', do_visualize=True)
