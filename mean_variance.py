import matplotlib.pyplot as plt
import numpy as np


class Portfolio:
    def __init__(self, risk_free_rate, mean, volatility, correlation, investment_horizon):
        self.mean = np.array(mean)

        if len(volatility) != len(mean):
            raise SystemExit("the length of volatility does not match the length of mean")
        else:
            self.volatility = np.array(volatility)

        self.risk_free_rate = np.array(risk_free_rate)

        if len(correlation) != len(mean):
            raise SystemExit("the length of correlation does not match the length of mean")
        else:
            self.correlation = np.array(correlation)
        if not np.allclose(self.correlation, self.correlation.transpose(), rtol=1e-05, atol=1e-08):
            raise SystemExit("the correlaiton matrix is not symmetric")

        self.investment_horizon = investment_horizon

        self.initial_stock_price = np.ones(self.mean.size)
        self.initial_wealthy = np.ones(1)

    @property
    def covariance(self):
        covariance = np.dot(np.dot(np.diag(self.volatility), self.correlation), np.diag(self.volatility))
        return covariance

    # @property
    def rel_risk_allocation(self):
        rel_risk_allocation = np.dot(np.linalg.inv(self.covariance), (self.mean - self.risk_free_rate)) / np.sum(
            np.dot(np.linalg.inv(self.covariance), (self.mean - self.risk_free_rate)))

        return rel_risk_allocation

    @property
    def excess_return(self):
        excess_return = np.dot((self.mean - self.risk_free_rate).transpose(), self.rel_risk_allocation())
        return excess_return

    @property
    def risk_port_volatility(self):
        risk_port_volatility = np.sqrt(
            np.dot(np.dot((self.mean - self.risk_free_rate).transpose(), np.linalg.inv(self.covariance)), (
                self.mean - self.risk_free_rate))) / np.abs(np.sum(
            np.dot(np.linalg.inv(self.covariance), (self.mean - self.risk_free_rate))))

        return risk_port_volatility

    @property
    def sharpe_ratio(self):
        sharpe_ratio = self.excess_return / self.risk_port_volatility
        return sharpe_ratio

    def set_risk_aversion(self, risk_aversion):
        self.risk_aversion = risk_aversion

    def risk_port_leverage(self):
        risk_port_leverage = 1 / (1 + self.risk_aversion) * (
            self.excess_return / np.dot(self.risk_port_volatility, self.risk_port_volatility))
        return risk_port_leverage

    # @property
    def abs_risk_port_allocation(self):
        abs_risk_port_allocation = 1 / (1 + self.risk_aversion) * np.dot(np.linalg.inv(self.covariance), (
            self.mean - self.risk_free_rate))
        # self.abs_risk_port_allocation = abs_risk_port_allocation

        return abs_risk_port_allocation

    def expected_total_mean(self):
        expected_total_mean = (self.risk_free_rate + 1 / (1 + self.risk_aversion) * (
            self.excess_return / self.risk_port_volatility) * (
                                   self.excess_return / self.risk_port_volatility) - 0.5 * 1 / (
                                   (1 + self.risk_aversion) * (1 + self.risk_aversion)) * (
                                   self.excess_return / self.risk_port_volatility) * (
                                   self.excess_return / self.risk_port_volatility)) * self.investment_horizon
        return expected_total_mean

    def expected_total_variance(self):
        expected_total_variance = self.investment_horizon / (
            (1 + self.risk_aversion) * (1 + self.risk_aversion)) * (
                                      self.excess_return / self.risk_port_volatility) * (
                                      self.excess_return / self.risk_port_volatility)

        return expected_total_variance

    def plot_mean_variance_frontier(self):
        # inverse_risk_aversion_vector = np.arange(-0.3, 100, 0.01)
        # risk_aversion_vector = 1 / inverse_risk_aversion_vector
        risk_aversion_vector = np.arange(-0.3, 50, 0.01)

        mean_frontier = []
        variance_frontier = []
        kelly_mean = 0
        kelly_variance = 0
        risk_free_mean = 0
        risk_free_variance = 0
        aversion_1_mean = 0
        aversion_minus_variance = 0
        aversion_minus_mean = 0
        aversion_1_variance = 0
        for item in risk_aversion_vector:
            # port = copy.deepcopy(self)
            self.set_risk_aversion(item)
            mean_frontier.append(self.expected_total_mean())
            variance_frontier.append(self.expected_total_variance())

            if np.absolute(item) < 0.01:
                kelly_mean = self.expected_total_mean()
                kelly_variance = self.expected_total_variance()
            elif item == np.max(risk_aversion_vector):
                risk_free_mean = self.expected_total_mean()
                risk_free_variance = self.expected_total_variance()

            elif np.absolute(item - 1) < 0.01:
                aversion_1_mean = self.expected_total_mean()
                aversion_1_variance = self.expected_total_variance()

            elif np.absolute(item + 0.2) < 0.01:
                aversion_minus_mean = self.expected_total_mean()
                aversion_minus_variance = self.expected_total_variance()

        plt.figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(variance_frontier, mean_frontier, 'k')
        plt.plot(kelly_variance, kelly_mean, 'ro')
        plt.annotate('Kelly($ \lambda=0$)', xy=(kelly_variance, kelly_mean),
                     xytext=(kelly_variance, kelly_mean - 0.05 * kelly_mean))

        plt.plot(risk_free_variance, risk_free_mean, 'ro')
        plt.annotate('Risk Free($ \lambda=+\infty$)', xy=(risk_free_variance, risk_free_mean),
                     xytext=(risk_free_variance + 0.01 , risk_free_mean))

        plt.plot(aversion_1_variance, aversion_1_mean, 'ro')
        plt.annotate('Half Kelly($ \lambda=1$)', xy=(aversion_1_variance, aversion_1_mean),
                     xytext=(aversion_1_variance + 0.1 * aversion_1_variance, aversion_1_mean))

        plt.plot(aversion_minus_variance, aversion_minus_mean, 'ro')
        plt.annotate('Over Bet($ \lambda=-0.2$)', xy=(aversion_minus_variance, aversion_minus_mean),
                     xytext=(aversion_minus_variance - 0.05 * aversion_minus_variance,
                             aversion_minus_mean - 0.05 * aversion_minus_mean))

        plt.title('Mean Variance Analysis')
        plt.xlabel('Variance of continuously compounding return')
        plt.ylabel('Mean of continuously compounding return')
        # plt.xticks([])
        # plt.yticks([])
        plt.subplots_adjust(left=0.15)
        plt.show()

    def set_initial_condition(self, initial_stock_price, initial_wealthy):
        self.initial_stock_price = np.array(initial_stock_price)
        self.initial_wealthy = np.array(initial_wealthy)

    def set_simulation_parameter(self, rebalance_time, number_path):
        self.rebalance_time = rebalance_time
        self.number_path = number_path

    def simulation_one_path(self):

        time_grid = np.arange(0, self.investment_horizon, self.rebalance_time)
        time_grid = np.append(time_grid, self.investment_horizon)

        mean = (self.mean - 0.5 * np.dot(self.volatility.transpose(), self.volatility))
        cov = self.covariance
        stock_price_evolution = np.array([self.initial_stock_price])
        wealthy_evolution = np.array([self.initial_wealthy])
        initial_stock_position = np.array(
            [(self.abs_risk_port_allocation() * self.initial_wealthy) / self.initial_stock_price])
        initial_bank_account = np.array([self.initial_wealthy * (1 - np.sum(self.abs_risk_port_allocation()))])
        stock_position_evolution = initial_stock_position
        bank_account_evolution = initial_bank_account

        pre_time = 0

        for time in time_grid[1:]:
            dt = time - pre_time

            random_return = np.random.multivariate_normal(mean * dt, cov * dt)
            # print(random_return)
            new_stock_price = stock_price_evolution[-1, :] * np.exp(random_return)
            new_wealthy = np.dot(stock_position_evolution[-1, :].transpose(), new_stock_price) + bank_account_evolution[
                                                                                                     -1] * np.exp(
                self.risk_free_rate * dt)
            new_stock_position = new_wealthy * self.abs_risk_port_allocation() / new_stock_price
            new_bank_account = new_wealthy * (1 - np.sum(self.abs_risk_port_allocation()))

            stock_price_evolution = np.vstack([stock_price_evolution, new_stock_price])
            stock_position_evolution = np.vstack([stock_position_evolution, new_stock_position])
            bank_account_evolution = np.vstack([bank_account_evolution, new_bank_account])
            wealthy_evolution = np.vstack([wealthy_evolution, new_wealthy])
            pre_time = time

        # print(wealthy_evolution)

        return stock_price_evolution, stock_position_evolution, bank_account_evolution, wealthy_evolution

    def simulation_analysis(self):

        final_wealthy_vector = []
        for num in range(self.number_path):
            final_wealthy = (self.simulation_one_path()[-1])[-1]

            final_wealthy_vector.append(final_wealthy)

        final_wealthy_vector = np.array(final_wealthy_vector)
        if np.min(final_wealthy_vector) <= 0:
            print("Warning: there are negative final wealthy")

        final_wealthy_vector[final_wealthy_vector < 0] = 0.001
        log_return = np.log(final_wealthy_vector / self.initial_wealthy)

        mean_return = np.mean(log_return)
        varance_return = np.var(log_return)
        return mean_return, varance_return

    def run_print_details(self):
        print("**************** PRINT DETAILS*******************")
        print("relative risk portfolio allocation is", self.rel_risk_allocation())
        print('sharpe ratio is', self.sharpe_ratio)
        print('excess return is', self.excess_return)
        print('risk portfolio leverage is', self.risk_port_leverage())
        print('total expected return is', self.expected_total_mean())
        print('total variance of return is', self.expected_total_variance())
        print('total volatility of return is', np.sqrt(self.expected_total_variance()))
        print('absolute risk portfolio position is', self.abs_risk_port_allocation())
        print('cash position is', 1 - np.sum(self.abs_risk_port_allocation()))
        print('simulated mean is', self.simulation_analysis()[0])
        print('simulated variance is', self.simulation_analysis()[1])


def main():
    risk_free_rate = 0.05
    mean = [0.10, 0.08]
    volatility = [0.1, 0.12]
    correlation = [[1, -0.3], [-0.3, 1]]

    investment_horizon = 1
    p = Portfolio(risk_free_rate, mean, volatility, correlation, investment_horizon)
    p.set_risk_aversion(1)

    rebalance_time = 0.01
    number_path = 1000
    p.set_simulation_parameter(rebalance_time, number_path)
    p.run_print_details()
    p.plot_mean_variance_frontier()

    x = 1


if __name__ == "__main__":
    main()