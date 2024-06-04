import sympy
import matplotlib.pyplot as plt
import sys

# Add the specific directory to the Python path
script_directory = '/Users/marcelgeller/PycharmProjects/curve_informing_simson/venv/Testings/curve_informing'
sys.path.append(script_directory)
from config import cfg


def baseline_scenario():
    prices = {}
    for year in range(2023, 2101):
        prices[year] = {'EAF': 150, 'BF/BOF': 400}
    return prices


def degree_1_5_scenario():
    prices = {}
    for year in range(2023, 2101):
        if year <= 2025:
            prices[year] = {'EAF': 150, 'BF/BOF': 400}
        elif 2025 < year <= 2035:
            bf_bof_price = 400 + (800 - 400) * (year - 2025) / (2035 - 2025)
            prices[year] = {'EAF': 150, 'BF/BOF': bf_bof_price}
        elif 2035 < year <= 2045:
            bf_bof_price = 800 - (800 - 600) * (year - 2035) / (2045 - 2035)
            prices[year] = {'EAF': 150, 'BF/BOF': bf_bof_price}
        else:
            prices[year] = {'EAF': 150, 'BF/BOF': 600}
    return prices


def prices_BF_BOF(scenario, year=None):
    """
    Print the BF/BOF prices of a specific scenario for each year or for a specific year.

    :param scenario: Dictionary containing prices for each year and the selected scenario.
    :param year: Specific year to print the price for. If None, print all years.
    """
    if year is not None:
        if year in scenario:
            print(f"Year {year}: BF/BOF Price = {scenario[year]['BF/BOF']} $/t")
            return scenario[year]['BF/BOF']
        else:
            print(f"Year {year} not found in the scenario.")
            return None
    else:
        for year in scenario:
            print(f"Year {year}: BF/BOF Price = {scenario[year]['BF/BOF']} $/t")
        return None


def prices_EAF(scenario, year=None):
    """
    Print the EAF prices of a specific scenario for each year or for a specific year.

    :param scenario: Dictionary containing prices for each year and the selected scenario.
    :param year: Specific year to print the price for. If None, print all years.
    """
    if year is not None:
        if year in scenario:
            print(f"Year {year}: EAF Price = {scenario[year]['EAF']} $/t")
            return scenario[year]['EAF']
        else:
            print(f"Year {year} not found in the scenario.")
            return None
    else:
        for year in scenario:
            print(f"Year {year}: EAF Price = {scenario[year]['EAF']} $/t")
        return None


def get_price_for_scenario_and_year(year=None):
    """
    Get the price for a given scenario and year based on the configuration.

    :param year: Year for which to get the price.
    :return: Tuple of (BF/BOF price, EAF price) or None if the year is not found.
    """
    scenario_type = cfg.price_scenario
    if scenario_type == 'baseline':
        scenario = baseline_scenario()
    elif scenario_type == '1_5_degree':
        scenario = degree_1_5_scenario()
    else:
        print("Unknown scenario type in config.")
        return None

    bf_bof_price = prices_BF_BOF(scenario, year)
    eaf_price = prices_EAF(scenario, year)
    return eaf_price, bf_bof_price


t_price = 2050
# Test
print("BF/BOF and EAF prices for", cfg.price_scenario, "scenario:")
eaf, bof = get_price_for_scenario_and_year(t_price)
print(eaf)
print(bof)


def main():
    def plot_scenarios():
        baseline_prices = baseline_scenario()
        degree_1_5_prices = degree_1_5_scenario()

        years = list(baseline_prices.keys())
        eaf_prices = [baseline_prices[year]['EAF'] for year in years]  # EAF prices are the same for both scenarios
        baseline_bf_bof = [baseline_prices[year]['BF/BOF'] for year in years]
        degree_1_5_bf_bof = [degree_1_5_prices[year]['BF/BOF'] for year in years]

        plt.figure(figsize=(14, 8))

        # Plot EAF Prices (common for both scenarios)
        plt.plot(years, eaf_prices, label='EAF Price - Baseline & 1.5 Degree Scenario', linestyle=':', color='grey')

        # Plot BF/BOF Prices for Baseline Scenario
        plt.plot(years, baseline_bf_bof, label='BF/BOF Price - Baseline Scenario', linestyle='--', color='orange')

        # Plot BF/BOF Prices for 1.5 Degree Scenario
        plt.plot(years, degree_1_5_bf_bof, label='BF/BOF Price - 1.5 Degree Scenario', linestyle='-', color='darkgreen')

        plt.xlabel('Year')
        plt.ylabel('Price (USD/t)')
        plt.title('EAF and BF/BOF Prices in Different Scenarios')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_scenarios()


if __name__ == "__main__":
    main()
