import statistics

import numpy as np
import pandas as pd
from src.predict.prediction_tools import split_future_stocks_to_base_year_categories, \
    copy_stocks_across_scenarios
from src.read_data.load_data import load_region_names_list
from src.tools.config import cfg
from darts.metrics import mape
from darts.models import RNNModel
from darts import TimeSeries
from matplotlib import pyplot as plt
import os
import datetime

do_stock_change = False
do_normalize_stocks = True
do_total_not_per_capita = False
do_use_population = False
n_epochs = 700
do_show_plot = False
do_create_new_model = True
input_chunk_length = 95  # 109-14 time steps for future
output_chunk_length = 14  # or 92 ?
n_rnn_layers = 8

cfg.n_epochs = n_epochs
cfg.n_rnn_layers = n_rnn_layers


def predict_lstm(stocks, gdp, pop, include_scenarios=False):  # todo change include scenarios to cfg.include_gdp...
    gdp = gdp[:, :, 1]  # only use SSP2 TODO change
    pop = pop[:, :, 1]

    past_stocks_by_category = stocks.copy()  # TODO dis dubble?
    stocks = np.sum(stocks, axis=2)

    future_stocks = _lstm_stock_curve(stocks, gdp, pop, do_create_new_model=do_create_new_model)
    if include_scenarios:
        future_stocks = np.moveaxis(future_stocks, 0, -1)

    future_stocks = split_future_stocks_to_base_year_categories(past_stocks_by_category, future_stocks,
                                                                is_future_stocks_with_scenarios=include_scenarios)
    if include_scenarios:
        past_stocks_by_category = copy_stocks_across_scenarios(past_stocks_by_category)

    stocks = np.concatenate([past_stocks_by_category, future_stocks], axis=0)

    return stocks


def _lstm_stock_curve(stocks, gdp, pop, do_create_new_model):
    if not do_create_new_model:
        model, model_mape = _load_model()
        if model is None:
            do_create_new_model = True
    if do_total_not_per_capita:
        stocks = np.einsum('tr,tr->tr', stocks, pop[:109])
        gdp = np.einsum('tr,tr->tr', gdp, pop)

    if do_stock_change:
        stock_change = np.zeros_like(stocks)
        stock_change[0] = stocks[0]
        stock_change[1:] = stocks[1:] - stocks[:-1]
        stocks = stock_change  # todo decide?
        stocks = np.maximum(0.001, stocks)

        gdp_change = np.zeros_like(gdp)
        gdp_change[0] = gdp[0]
        gdp_change[1:] = gdp[1:] - gdp[:-1]
        gdp = gdp_change
        gdp = np.maximum(0, gdp)

    if do_normalize_stocks:
        min_stocks = np.min(stocks, axis=0)
        max_stocks = np.max(stocks, axis=0)
        max_stocks = np.ones_like(min_stocks) * 40  # TODO explain what max 40 t/cap
        stocks = (stocks - min_stocks) / (max_stocks - min_stocks)

        min_gdp = np.min(gdp, axis=0)
        max_gdp = np.max(gdp, axis=0)
        gdp = (gdp - min_gdp) / (max_gdp - min_gdp)

    stock_times = pd.date_range("1900-01-01", periods=109, freq="Y")
    gdp_times = pd.date_range("1900-01-01", periods=201, freq="Y")

    regions = load_region_names_list()

    ts_stocks = TimeSeries.from_times_and_values(stock_times, stocks)
    ts_gdp = TimeSeries.from_times_and_values(gdp_times, gdp)
    ts_stocks_train, _ = ts_stocks.split_before(0.8)
    ts_stocks_list = [TimeSeries.from_times_and_values(stock_times, stocks[:, r]) for r, region in enumerate(regions)]

    ts_gdp_list = [TimeSeries.from_times_and_values(gdp_times, gdp[:, r]) for r, region in enumerate(regions)]
    ts_pop_and_gdp_list = [TimeSeries.from_times_and_values(gdp_times, np.array([pop[:, r], gdp[:, r]]).transpose())
                           for r, region in enumerate(regions)]

    for i in range(0):  # TODO delete?
        ts_stocks_list.append(ts_stocks_list[11])
        ts_stocks_list.append(ts_stocks_list[2])
        ts_gdp_list.append(ts_gdp_list[11])
        ts_gdp_list.append(ts_gdp_list[2])

    ts_stocks_test = TimeSeries.from_times_and_values(stock_times, stocks[:, 0])
    ts_gdp_test = TimeSeries.from_times_and_values(gdp_times, gdp[:, 0])

    ts_covariates = ts_pop_and_gdp_list if do_use_population else ts_gdp_list
    if do_create_new_model:
        ts_stocks_list_without_ref = ts_stocks_list[:9] + ts_stocks_list[9 + 1:]
        ts_covariates_without_ref = ts_covariates[:9] + ts_covariates[9 + 1:]
        model, model_mape = _create_new_model(
            ts_stocks_list_without_ref if cfg.model_type == 'inflow' else ts_stocks_list,
            ts_covariates_without_ref if cfg.model_type == 'inflow' else ts_covariates)

    prediction = model.predict(n=92,
                               series=ts_stocks_list,
                               future_covariates=ts_pop_and_gdp_list if do_use_population else ts_gdp_list)
    values = np.array([prediction[r].values() for r, region in enumerate(regions)]).transpose()
    # stocks = stocks[:, 0]
    # values = values[:, 0]
    smoothed_times, orig_times = _smooth_stocks(stocks, values[0])
    if do_normalize_stocks:
        smoothed_stocks = smoothed_times * (max_stocks - min_stocks) + min_stocks
        orig_stocks = orig_times * (max_stocks - min_stocks) + min_stocks
    else:
        smoothed_stocks = smoothed_times
        orig_stocks = orig_times

    if do_stock_change:
        smoothed_stocks = np.cumsum(smoothed_stocks, axis=0)
        orig_stocks = np.cumsum(orig_stocks, axis=0)

    if do_total_not_per_capita:
        a = 0  # todo delete
        # orig_stocks = np.einsum('tr,tr->tr', orig_stocks, 1 / pop) # todo uncomment?
        # gdp = np.einsum('tr,tr->tr', gdp, pop, 1 / pop)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    f_name = f'{model_mape:.2f}%_{timestamp}_{cfg.steel_data_source}_{cfg.region_data_source}_' \
             f'{cfg.model_type}_{cfg.n_epochs}_{cfg.n_rnn_layers}_{cfg.hidden_dim}_{input_chunk_length}_{output_chunk_length}'
    base_path = os.path.join(cfg.data_path, 'models', 'lstm_models')
    pic_path = os.path.join(base_path, f'{f_name}.png')
    model_path = os.path.join(base_path, f'{f_name}.pt')
    if do_create_new_model or do_show_plot:
        plt.plot(np.arange(1900, 2101), orig_stocks)
        plt.plot(np.arange(2000, 2101), smoothed_stocks[100:], '--')
        # plt.plot(np.arange(1900, 2101), timelines[:, 0])
        # plt.plot(np.arange(1900, 2101), gdp[:, 0])
        plt.legend(regions)
        plt.xlabel('Time (y)')
        plt.ylabel('Steel (t)')
        plt.title('Steel stocks per capita with normal and smoothed (--) predictions \n'
                  f'{cfg.region_data_source}_{cfg.steel_data_source}_{cfg.model_type}_{cfg.n_epochs}_{cfg.n_rnn_layers}_{cfg.hidden_dim}')
    if do_create_new_model:
        model.save(model_path)
        plt.savefig(pic_path, dpi=300)
    if do_show_plot:
        plt.show()
    plt.clf()  # clear figure if doing several tests #todo implement that it works this file and first_matrix_test
    return orig_stocks[109:]  # smoothed_stocks[109:]  # todo decide this (basically orig_stocks) vs smoothed version


def _load_model():
    base_path = os.path.join(cfg.data_path, 'models', 'lstm_models')
    files = os.listdir(base_path)
    models = [file.split('_') for file in files if file.endswith('.pt')]
    models = [model for model in models if
              model[3] == cfg.steel_data_source and model[4] == cfg.region_data_source and model[
                  5] == cfg.model_type and float(model[0][:-1]) < 10]
    # model needs to match desired steel and region data source and model type and have an accuracy of at least 10 %

    if len(models) == 0:
        return None, None
    models = ['_'.join(model) for model in models]
    final = min(models)
    final_path = os.path.join(base_path, final)
    print(final_path)
    model_mape = float(final.split('%')[0])
    model = RNNModel.load(final_path)
    return model, model_mape


def _create_new_model(ts_stocks_list, ts_covariates):
    print(
        f'Creating new model with {cfg.n_rnn_layers} rnn layers, {cfg.hidden_dim} hidden dim size and {cfg.n_epochs} epochs.')
    model = RNNModel(model='LSTM',
                     input_chunk_length=input_chunk_length,
                     output_chunk_length=output_chunk_length,
                     n_rnn_layers=cfg.n_rnn_layers)
    model.fit(ts_stocks_list,
              future_covariates=ts_covariates,
              epochs=cfg.n_epochs,
              verbose=True)
    model_mape = eval_model(model, ts_stocks_list,
                            future_covariates=ts_covariates)

    return model, model_mape


def _smooth_stocks(stocks, predicted):
    # return np.concatenate([stocks, predicted], axis=0)
    historic_stock_change = stocks[1:] - stocks[:-1]
    historic_stock_change_change = historic_stock_change[1:] - historic_stock_change[:-1]
    last_stock = stocks[-1]
    last_stock_change = historic_stock_change[-1]
    max_stock_change_change = np.percentile(np.abs(historic_stock_change_change[50:]),
                                            99)  # only consider more recent (since 1950), 80th percentile
    min_stock_change = np.percentile(historic_stock_change, 1)
    max_stock_change = np.percentile(historic_stock_change, 99)
    new_predicted = np.zeros_like(predicted)

    for y, predicted_stock in enumerate(predicted):
        wanted_stock = (last_stock + predicted_stock) / 2
        wanted_stock_change = wanted_stock - last_stock
        wanted_stock_change_change = wanted_stock_change - last_stock_change

        new_stock_change_change = wanted_stock_change_change.copy()
        new_stock_change_change[np.abs(new_stock_change_change) > max_stock_change_change] = \
            (np.sign(new_stock_change_change) * max_stock_change_change)[
                np.abs(new_stock_change_change) > max_stock_change_change]

        new_stock_change = new_stock_change_change + last_stock_change
        new_stock_change[new_stock_change < min_stock_change] = min_stock_change
        new_stock_change[new_stock_change > max_stock_change] = max_stock_change
        new_predicted_stock = new_stock_change + last_stock

        new_predicted[y] = new_predicted_stock

        last_stock = new_predicted_stock
        last_stock_change = new_stock_change

    orig_stocks = np.concatenate([stocks, predicted], axis=0)
    smoothed_stocks = np.concatenate([stocks, new_predicted], axis=0)

    return smoothed_stocks, orig_stocks


def eval_model(model, stocks, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=stocks,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=95,
                                          # TODO 2008-(2022-2009)=1995))
                                          retrain=False,
                                          verbose=True)

    # stocks[-len(backtest) - 100:].plot()
    # backtest.plot(label='backtest (n=10)')
    mape_list = mape(stocks, backtest)
    average_mape = statistics.fmean(mape_list)
    print('Backtest MAPE = {}'.format(average_mape))
    return average_mape


if __name__ == '__main__':
    normal = False  # todo delete structure?
    from src.predict.calc_steel_stocks import test

    if normal:
        test(strategy='LSTM', do_visualize=True)
    else:
        region_tests = ['REMIND']  # Options: ['Pauliuk', 'REMIND']
        data_tests = ['IEDatabase']  # Options: ['Mueller', 'IEDatabase', 'ScrapAge']
        # TODO decide: does it make sense to make specific models for different approaches
        model_types = ['stock', 'inflow', 'change']
        n_epochen = [1000]
        n_rnn_layers = [3, 6, 9, 12]
        hidden_dims = [5, 15, 25, 35]
        for region in region_tests:
            cfg.region_data_source = region
            for data_set in data_tests:
                cfg.steel_data_source = data_set
                for model_type in model_types:
                    cfg.model_type = model_type
                    for n_epoch in n_epochen:
                        cfg.n_epochs = n_epoch
                        for n_rnn in n_rnn_layers:
                            cfg.n_rnn_layers = n_rnn
                            for hidden_dim in hidden_dims:
                                cfg.hidden_dim = hidden_dim
                                print(f'\n\nTest {region} {data_set} {model_type} {n_epoch} {n_rnn} {hidden_dim}\n\n')
                                test(strategy='LSTM', do_visualize=False)
