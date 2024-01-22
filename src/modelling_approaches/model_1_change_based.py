import numpy as np
from math import e, pi, sqrt
from src.base_model.load_params import get_cullen_fabrication_yield
from src.modelling_approaches.load_model_dsms import load_model_dsms, get_dsm_data
from src.modelling_approaches.load_data_for_approaches import get_past_production_trade_forming_fabrication, \
    get_past_stocks
from src.odym_extension.MultiDim_DynamicStockModel import MultiDim_DynamicStockModel


def get_change_based_model_upper_cycle(country_specific=False):  # TODO same as model 3 -> make more efficient?
    production, trade, forming_fabrication, indirect_trade = \
        get_past_production_trade_forming_fabrication(country_specific)

    past_dsms = load_model_dsms(country_specific=country_specific,
                                do_past_not_future=True,
                                model_type='change',
                                do_econ_model=False,
                                recalculate=False,  # TODO change - still some weird erros
                                forming_fabrication=forming_fabrication,
                                indirect_trade=indirect_trade)  # TODO change recalculate
    inflows, stocks, outflows = get_dsm_data(past_dsms)
    fabrication_use = inflows - indirect_trade

    return production, trade, forming_fabrication, fabrication_use, indirect_trade, inflows, stocks, outflows


def _calc_stock_change(stocks):  # TODO: Unnecessary? -> nope!
    stock_change = np.zeros_like(stocks)
    stock_change[0] = stocks[0]
    stock_change[1:] = np.diff(stocks, axis=0)
    return stock_change


def _calc_fabrication_by_category(forming_fabrication, stock_change, indirect_trade, fabrication_yield):
    """
    For Model approach 1 (change-based), we assume that inflow category share is equal to direct stock change category
    share (where direct stock change and forming_fabrication-fabrication flow are positive, non-zero, otherwise 0).
    Direct stock change is defined as the stock change minus the indirect trade.
    This has inherent lifetime assumptions, but is the best approach available.
    Split is assumed to be 25% for all categories if sum of direct stock change is 0.
    # TODO create LaTeX explanation?
    :param forming_fabrication:
    :param stock_change:
    :param indirect_trade:
    :param fabrication_yield:
    :return:
    """
    direct_stock_change = stock_change - indirect_trade
    positive_direct_stock_change = np.maximum(0, direct_stock_change)
    sum_positive_direct_stock_change = np.expand_dims(np.sum(positive_direct_stock_change, axis=2), axis=2)
    direct_stock_change_category_split = np.divide(positive_direct_stock_change, sum_positive_direct_stock_change,
                                                   out=np.ones_like(positive_direct_stock_change) * 0.25,
                                                   where=sum_positive_direct_stock_change != 0)

    split_yield_matrix = np.einsum('g,trd->trdg', fabrication_yield, direct_stock_change_category_split)
    swapped_split_yield_matrix = np.swapaxes(split_yield_matrix, 2, 3)
    a_matrix = np.divide(split_yield_matrix, swapped_split_yield_matrix,
                         out=np.zeros_like(split_yield_matrix),
                         where=swapped_split_yield_matrix != 0)
    a_matrix = np.sum(a_matrix, axis=2)

    fabrication_category_split = np.divide(1, a_matrix, out=np.zeros_like(a_matrix), where=a_matrix != 0)
    fabrication_by_category = np.einsum('tr,trg->trg', forming_fabrication, fabrication_category_split)

    return fabrication_by_category


def get_change_based_past_dsms(country_specific, fabrication=None, indirect_trade=None):
    if fabrication is None or indirect_trade is None:
        production, trade, forming_fabrication, indirect_trade = \
            get_past_production_trade_forming_fabrication(country_specific)
        fabrication = forming_fabrication

    stocks = get_past_stocks(country_specific)
    stock_change = _calc_stock_change(stocks)
    fabrication_yield = get_cullen_fabrication_yield()
    fabrication_by_category = _calc_fabrication_by_category(fabrication, stock_change, indirect_trade,
                                                            fabrication_yield)
    fabrication_use = np.einsum('g,trg->trg', fabrication_yield, fabrication_by_category)

    inflows = fabrication_use + indirect_trade
    outflows = inflows - stock_change

    # TODO: delete test functions
    test_outflow = np.sum(np.sum(outflows, axis=1), axis=1)
    test_outflow2 = test_outflow > 0
    test_outflow3 = np.all(test_outflow2)

    lifetime_mean = _calc_lifetimes(inflows, outflows)

    inflows = np.moveaxis(inflows, 0, 2)  # move time axis to the end to iterate more easily through inflows

    years = np.arange(1900, 2009)  # TODO: Change all numbers
    dsms = [[_create_inflow_based_past_dsm(cat_inflows,
                                           stocks[:, region_idx, cat_idx],
                                           outflows[:, region_idx, cat_idx],
                                           lifetime_mean[:, region_idx, cat_idx],
                                           years)
             for cat_idx, cat_inflows in enumerate(region_inflows)]
            for region_idx, region_inflows in enumerate(inflows)]

    return dsms


def _calc_lifetimes(inflows, outflows):
    mean = np.zeros_like(inflows)
    outflows[outflows == 0] = 1  # assume some small outflows  # TODO decide / write down
    mean[0] = _calc_mean_t(inflows[0], outflows[0], outflow_past_cohorts=0)  #

    lifetime_matrix_shape = (outflows.shape[0],) + outflows.shape
    lifetime_matrix = np.zeros(lifetime_matrix_shape)

    _update_lifetime_matrix(lifetime_matrix, mean, 0)
    # TODO something fishy here --> mean seems to be way to low,
    #  -> seems to be due to too high stock change from stock data

    for t in range(1, 109):  # TODO change 109
        outflow_past_cohorts = inflows[:t] * lifetime_matrix[t, :t]
        outflow_past_cohorts = np.sum(outflow_past_cohorts, axis=0)
        mean[t] = _calc_mean_t(inflows[t], outflows[t], outflow_past_cohorts=outflow_past_cohorts)
        _update_lifetime_matrix(lifetime_matrix, mean, t)

    return mean


def _update_lifetime_matrix(lifetime_matrix, mean, t):
    mean_t = mean[t]
    # mean_t[:] = 30  # todo: delete
    sd_t = 0.3 * mean_t
    future_t = np.arange(t, 109)  # TODO change
    t_dash = t

    factor = 1 / (sqrt(2) * pi * sd_t)
    prep_exponent_divident = np.expand_dims(np.expand_dims(future_t - t_dash, axis=1), axis=1)
    exponent_divident = - (prep_exponent_divident - mean_t) ** 2
    exponent_divisor = 2 * sd_t ** 2
    exponent = np.einsum('trg,rg->trg', exponent_divident, 1 / exponent_divisor)

    new_lifetimes = np.einsum('rg,trg->trg', factor, e ** exponent)
    new_lifetimes = np.nan_to_num(new_lifetimes)
    lifetime_matrix[t:, t] = new_lifetimes


def _calc_mean_t(inflow_t, outflow_t, outflow_past_cohorts):
    mean_t_divisor = ((outflow_t - outflow_past_cohorts) * 0.3 * sqrt(2) * pi * e ** (1 / (2 * (0.3) ** 2)))
    mean_t = inflow_t / mean_t_divisor
    return mean_t
    # TODO: decide?np.divide(inflow_t, mean_0_divisor, out=np.ones_like(inflow_t) * np.inf, where=outflow_t != 0)


def _create_inflow_based_past_dsm(inflows, stocks, outflows, lifetime_mean, years):
    steel_stock_dsm = MultiDim_DynamicStockModel(t=years,
                                                 i=inflows,
                                                 s=stocks,
                                                 o=outflows,
                                                 lt={'Type': 'Normal', 'Mean': lifetime_mean,
                                                     'StdDev': lifetime_mean * 0.3})  # TODO write down assumption

    # Optional - TODO: decide if necessary
    steel_stock_dsm.compute_s_c_inflow_driven()
    steel_stock_dsm.compute_o_c_from_s_c()
    return steel_stock_dsm


if __name__ == '__main__':
    from src.modelling_approaches.compute_upper_cycle import test

    test(model_type='change')
