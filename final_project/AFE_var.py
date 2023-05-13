import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as sps
from arch import arch_model
import yfinance as yf

### VAR MODEL FUNCTIONS

## Returns is a numpy array
def Non_parametric_VaR(time,returns,days = 252,alpha = 0.05):
    #percent = -np.percentile(returns.values[time-days:time]/100, alpha*100, interpolation='lower')
    scene = np.sort(returns[time-days:time]/100)
    VaR = -scene[math.floor(days*(alpha))]
    ES = -np.mean(scene[0:math.floor(days*(alpha))])
    return [VaR*100,ES*100]


#Either send Probability weights as an argument to the function or create inside the function
#Better to send so that u can create different functions which generate different prob weights

#This returns geometric decay weights
def prob_weights1(n = 252,l = 0.995):
    return np.array([((1 - l)*l**(n-i))/(1 - l**n) for i in range(1, n+1)])
    
def Non_parametric_VaR_prob(time,returns,days=252,alpha=0.05):
    PW = prob_weights1(days)
    scene = returns[time-days:time]
    data = pd.DataFrame({'Weights': PW, 'Accumulated Weights': PW, 'Scenarios': scene/100})
    data = data.sort_values('Scenarios', ascending=True)
    data['Accumulated Weights'] = np.cumsum(data['Weights'])
    
    idx = np.where(data['Accumulated Weights'] > alpha)[0][0]
    VaR = -data['Scenarios'].values[idx]
    ES =-((data['Scenarios'].values[:idx-1]* data['Weights'].values[:idx-1]).sum() + (alpha - data['Accumulated Weights'].values[idx-1]) * data['Weights'].values[idx])/alpha
    return [VaR*100,ES*100]

def Non_parametric_VaR_Vol(time,returns,avg_ret,N,days=252,alpha=0.05,lamda=0.9):
    ## Start offset will used to decide the starting day for the EWMA here say its 200 note it has to be smaller than Window size
    start_off = 201
    init_ret = returns[:,(time-days)-start_off].reshape(-1,1)
    #print(init_ret)
    Def_Cov = np.dot(init_ret,init_ret.T)
    #print(Def_Cov.shape)
    #var_store = []
    #var_store.append(np.sum(Def_Cov)/(N*N))
    for i in range((time-days)-start_off+1,time-days-1):
        temp_ret = returns[:,i].reshape(-1,1)
        Def_Cov  = lamda*Def_Cov+(1-lamda)*np.dot(temp_ret,temp_ret.T)
    #print(Def_Cov)
    var_ewma = np.zeros((days + 1, 1))
    var_ewma[0,0] = np.sum(Def_Cov)/(N*N)
    for i in range(0,days):
        temp_ret = returns[:,time-days+i].reshape(-1,1)
        Def_Cov = lamda * Def_Cov + (1 - lamda) * np.dot(temp_ret,temp_ret.T)
        var_ewma[i+1,0] = np.sum(Def_Cov)/(N*N)
    #print(var_ewma[-1])    
    vol_ewma = var_ewma**0.5
    scale = vol_ewma[-1]/vol_ewma
    adj_ret = scale[:-1]*avg_ret[time-days:time]
    scene = np.sort(adj_ret)
    VaR =  -np.percentile(adj_ret, math.floor(alpha*100), interpolation='lower')
    ES = -np.mean(scene[0:math.floor(days*(alpha))])
    return [VaR*100,ES*100]

## This function considers it like a single stock so its single variate analysis
def Param_VaR_Normal_Sing(time,returns,days=252,alpha = 0.05):
    #Assuming Mean returns as 0 
    mean_return = 0
    vol = np.std(returns.values[time-days:time]) # this is the standard deviation of daily returns
    VaR = -vol*sps.norm.ppf(alpha, loc=mean_return, scale=1)
    ES = vol*(sps.norm.pdf(sps.norm.ppf(alpha, loc=mean_return, scale=1)))/(alpha)
    return [VaR,ES]

#Note return_arr this time takes in a numpy array
def Param_VaR_Normal_Mult(time,return_arr,N=1,days=252,alpha=0.05):
    #Assuming Mult_var Mean is zero
    #N is no of stocks or dim 0 of return_arr
    #print(return_arr.shape)
    Cov_Mat = np.cov(return_arr[:,time-days:time])
    vol = np.sqrt(np.sum(Cov_Mat)/(N*N))
    VaR = -vol*sps.norm.ppf(alpha, loc=0, scale=1)
    ES = vol*(sps.norm.pdf(sps.norm.ppf(alpha, loc=0, scale=1)))/(alpha)
    return [VaR,ES]

def Param_EWMA(time,return_arr,N=1,days=252,alpha = 0.05,lamda = 0.95):
    init_ret = return_arr[:,time-days].reshape(-1,1)
    Def_Cov = np.dot(init_ret,init_ret.T)
    for i in range(time-days+1,time):
        temp_ret = return_arr[:,i].reshape(-1,1)
        Def_Cov  = lamda*Def_Cov+(1-lamda)*np.dot(temp_ret,temp_ret.T)
    vol = np.sqrt(np.sum(Def_Cov)/(N*N))
    mean_return = 0
    VaR = -vol*sps.norm.ppf(alpha, loc=mean_return, scale=1)
    ES = vol*(sps.norm.pdf(sps.norm.ppf(alpha, loc=mean_return, scale=1)))/(alpha)
    return [VaR,ES]

def Garch_VaR(time,returns,days=252,alpha = 0.05):
    garch = arch_model(returns[time-days:time]/100, vol='garch', p=1, o=0, q=1)
    garch_fitted = garch.fit()
    garch_forecast = garch_fitted.forecast(horizon=1, reindex=False)
    variance = garch_forecast.variance.values[0][0]
    
    volatility = np.sqrt(variance)
    VaR = -volatility*sps.norm.ppf(alpha, loc=mean_return, scale=1)
    ES = volatility*(sps.norm.pdf(sps.norm.ppf(alpha, loc=mean_return, scale=1)))/(alpha)
    return [VaR*100,ES*100]





START = "2012-01-01"
END = "2022-01-01"
INTERVAL = "1d" # Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo and 3mo (m refers to minute, h refers to hour, d refers to day, wk refers to week and mo refers to month)

TICKERS = [
    "AMZN", # Amazon (CONSUMER DISCRETIONARY)
    "BMY",
    "TWI"
]

stock_data = yf.download(
    tickers = TICKERS,
    start = START,
    end = END,
    interval = INTERVAL
).dropna()['Adj Close']


col_names = list(stock_data.columns)
N = len(col_names)
store = np.zeros(stock_data[col_names[0]].shape)
for name in col_names:
    store+= np.array(stock_data[name])
store = store/N

avg_price_Series = pd.Series(store)
avg_price_returns = avg_price_Series.pct_change()*100


returns = []
for name in col_names:
    returns.append(np.array(stock_data[name].pct_change()*100))
### Returns of all stocks fed as numpy-ndarray 
return_arr = np.array(returns)