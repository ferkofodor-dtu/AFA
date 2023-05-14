import math
import numpy as np
import matplotlib.pyplot as plt


class EuropeanOption:
    def __init__(self, K, S0, T, r, sigma, N):
        self.dt = T/N
        self.K = K
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.R = np.exp(r * self.dt)

    def __int_prms__(self):
        self.u = math.exp(self.sigma*math.sqrt(self.dt))
        self.d = 1/self.u
        self.p = (self.R - self.d)/(self.u - self.d)
        ## (math.exp((self.r-self.q)*self.dt)-self.d)/(self.u-self.d)
        ## ((R + 1) - self.d)/(self.u - self.d)
        
#     def stocktree(self):
#         stocktree = [[0.0 for j in range(i+1)] for i in range(self.N+1)]
#         for i in range(self.N+1):
#             for j in range(i+1):
#                 stocktree[i][j] = self.S0*(self.u**(i-j))*(self.d**j)
#         return stocktree
    
    def stocktree(self, plot=True):
        stocktree = [[0.0 for j in range(i+1)] for i in range(self.N+1)]
        if plot:
            fig, ax = plt.subplots()
        for i in range(self.N+1):
            for j in range(i+1):
                stocktree[i][j] = self.S0*(self.u**(i-j))*(self.d**j)
                if plot:
                    ax.plot([i], [stocktree[i][j]], 'o', markersize=10, color='blue')
                    plt.text(i+0.02, stocktree[i][j]+0.02, f"{stocktree[i][j]:.2f}", fontsize=8)
        #if i < self.N:
        #    ax.plot([i, i+1], [stocktree[i][j], stocktree[i+1][j]], color='gray', linestyle='-', linewidth=2)
        #    ax.plot([i, i+1], [stocktree[i][j], stocktree[i+1][j+1]], color='gray', linestyle='-', linewidth=2)
        if plot:
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.set_title('Stock Option Tree')
            plt.show()
        return stocktree

    def option_price(self, stocktree, verbose=False):
        option = [[0.0 for j in range(i+1)] for i in range(self.N+1)]
        if self.is_call:
            for j in range(self.N+1):
                option[self.N][j] = max(0, stocktree[self.N][j]-self.K)
                if verbose:
                    print(option)
        else:
            for j in range(self.N+1):
                option[self.N][j] = max(0, self.K-stocktree[self.N][j])
                if verbose:
                    print(option)
        return option

    # def returntree(self, option, stocktree):
    #     for i in range(self.N-1, -1, -1):
    #         for j in range(i+1):
    #             if self.is_call:
    #                 option[i][j] = max(stocktree[i][j]-self.K, 0)
    #             else:
    #                 option[i][j] = max(self.K-stocktree[i][j], 0)
    #     return option

    def optpricetree(self, option, stocktree):
        for i in range(self.N-1, -1, -1):
            for j in range(i+1):
                if self.is_call:
                    option[i][j] = (self.p*option[i+1][j]+(1-self.p)*option[i+1][j+1])/self.R
                else:
                    option[i][j] = (self.p*option[i+1][j]+(1-self.p)*option[i+1][j+1])/self.R
        return option

    def price(self, is_call, plot=True, verbose=False):
        self.is_call = is_call
        self.__int_prms__()
        stocktree = self.stocktree(plot=plot)
        option = self.option_price(stocktree, verbose=verbose)
        return self.optpricetree(option, stocktree)[0][0]
    



import math
import numpy as np
import matplotlib.pyplot as plt

class OptionPricer:
    def __init__(self, S, K, T, r, sigma, N, is_american=True):
        self.dt = T/N
        self.K = K
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.R = np.exp(r * self.dt)
        self.is_american = is_american

    def __int_prms__(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (self.R - self.d) / (self.u - self.d)

    def stocktree(self, plot=True):
        stocktree = [[0.0 for j in range(i + 1)] for i in range(self.N + 1)]
        if plot:
            fig, ax = plt.subplots()
        for i in range(self.N + 1):
            for j in range(i + 1):
                stocktree[i][j] = self.S * (self.u ** (i - j)) * (self.d ** j)
                if plot:
                    ax.plot([i], [stocktree[i][j]], 'o', markersize=10, color='blue')
                    plt.text(i + 0.02, stocktree[i][j] + 0.02, f"{stocktree[i][j]:.2f}", fontsize=8)
        if plot:
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.set_title('Stock Option Tree')
            plt.show()
        return stocktree

    def option_price_american(self, stocktree, verbose=False):
        option = [[0.0 for j in range(i + 1)] for i in range(self.N + 1)]
        if self.is_call:
            for j in range(self.N + 1):
                option[self.N][j] = max(0, stocktree[self.N][j] - self.K)
                if verbose:
                    print(option)
        else:
            for j in range(self.N + 1):
                option[self.N][j] = max(0, self.K - stocktree[self.N][j])
                if verbose:
                    print(option)
        return option
    
    def option_price_american(self, stocktree, verbose=False):
        optiontree = [[0.0 for j in range(i + 1)] for i in range(self.N + 1)]

        for i in range(self.N + 1):
            for j in range(i + 1):
                stocktree[i][j] = self.S * (self.u ** (i - j)) * (self.d ** j)
                if i == self.N:
                    if self.is_call:
                        optiontree[i][j] = max(0, stocktree[i][j] - self.K)
                    else:
                        optiontree[i][j] = max(0, self.K - stocktree[i][j])

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                if self.is_call:
                    early_exercise = max(stocktree[i][j] - self.K, 0)
                else:
                    early_exercise = max(self.K - stocktree[i][j], 0)

                optiontree[i][j] = (self.p * optiontree[i + 1][j] + (1 - self.p) * optiontree[i + 1][j + 1]) * math.exp(-self.r * self.dt)
                optiontree[i][j] = max(early_exercise, optiontree[i][j])
        
        return optiontree
    
    def calculate_price_european(self, stocktree, verbose=False):
        optiontree = [[0.0 for j in range(i + 1)] for i in range(self.N + 1)]

        for i in range(self.N + 1):
            for j in range(i + 1):
                stocktree[i][j] = self.S * (self.u ** (i - j)) * (self.d ** j)
                if i == self.N:
                    if self.is_call:
                        optiontree[i][j] = max(0, stocktree[i][j] - self.K)
                    else:
                        optiontree[i][j] = max(0, self.K - stocktree[i][j])

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                optiontree[i][j] = (self.p * optiontree[i + 1][j] + (1 - self.p) * optiontree[i + 1][j + 1]) * math.exp(-self.r * self.dt)

        return optiontree

    def price(self, is_call, plot=True, verbose=False):
        self.is_call = is_call
        self.__int_prms__()
        stocktree = self.stocktree(plot=plot)
        if self.is_american:
            option = self.option_price_american(stocktree, verbose=verbose)
        else:
            option = self.calculate_price_european(stocktree, verbose=verbose)

        return option[0][0]




import math

def calculate_option_price_US(S, K, r, sigma, T, N, is_call=True):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    stocktree = [[0.0 for j in range(i + 1)] for i in range(N + 1)]
    optiontree = [[0.0 for j in range(i + 1)] for i in range(N + 1)]

    for i in range(N + 1):
        for j in range(i + 1):
            stocktree[i][j] = S * (u ** (i - j)) * (d ** j)
            if i == N:
                if is_call:
                    optiontree[i][j] = max(0, stocktree[i][j] - K)
                else:
                    optiontree[i][j] = max(0, K - stocktree[i][j])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            if is_call:
                early_exercise = max(stocktree[i][j] - K, 0)
            else:
                early_exercise = max(K - stocktree[i][j], 0)

            optiontree[i][j] = (p * optiontree[i + 1][j] + (1 - p) * optiontree[i + 1][j + 1]) * math.exp(-r * dt)
            optiontree[i][j] = max(early_exercise, optiontree[i][j])

    return optiontree[0][0]


def calculate_european_option_price(S, K, r, sigma, T, N, is_call=True):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    stocktree = [[0.0 for j in range(i + 1)] for i in range(N + 1)]
    optiontree = [[0.0 for j in range(i + 1)] for i in range(N + 1)]

    for i in range(N + 1):
        for j in range(i + 1):
            stocktree[i][j] = S * (u ** (i - j)) * (d ** j)
            if i == N:
                if is_call:
                    optiontree[i][j] = max(0, stocktree[i][j] - K)
                else:
                    optiontree[i][j] = max(0, K - stocktree[i][j])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            optiontree[i][j] = (p * optiontree[i + 1][j] + (1 - p) * optiontree[i + 1][j + 1]) * math.exp(-r * dt)

    return optiontree[0][0]
