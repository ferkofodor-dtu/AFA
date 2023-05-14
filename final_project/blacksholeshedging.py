import pandas as pd
import numpy as np
from scipy.stats import norm 
from math import log, sqrt, exp
import matplotlib.pyplot as plt

def black_scholes_call(S, K, r, t, sigma):
    d1 = (log(S / K) + (r + sigma**2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    call_price = S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, r, t, sigma):
    d1 = (log(S / K) + (r + sigma**2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    put_price = K * exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def BlackScholes(S_t = 40, K = 45, t = 0, T = 1, sigma = 1, r = 0.05, direction = "Call"):

    d1 = (1/(sigma*np.sqrt(T - t))) * (np.log(S_t/K) + (r + (sigma**2)/2)*(T - t))
    d2 = d1 - sigma*np.sqrt(T - t)
    
    if direction == "Call":  
        return(norm.cdf(d1)*S_t - norm.cdf(d2)*K*np.exp(-r*(T-t)))
    elif direction == "Put":
        return(-norm.cdf(-d1)*S_t + norm.cdf(-d2)*K*np.exp(-r*(T-t)))
    else:
        return(np.nan)
    
    

def black_scholes_delta(s0, k, r, t, sigma):
  d1 = (np.log(s0/k) + (r + 0.5*sigma**2)*t) / (sigma * np.sqrt(t))
  d2 = d1 - sigma * np.sqrt(t)
  N_d1 = norm.cdf(d1)
  N_d2 = norm.cdf(d2)
  delta = N_d1
  return delta


def return_frequency_interest(stock_prices, option_prices):
  rebalancing_frequency = 1/len(stock_prices)
  risk_free_interest = 0.05
  time_to_expiration = len(stock_prices)
  return rebalancing_frequency, risk_free_interest, time_to_expiration


def get_hedge_position(call_price, delta):
  hedge_position = call_price*(-delta)
  return hedge_position


def hedge_rebalancing(stock_prices, delta, call_prices, dt, k, r, sigma):
  current_stock_price = 0
  final_profit = 0
  hedge_position = get_hedge_position(call_prices[0], delta)
  hedges = []
  hedges.append(hedge_position)
  for i in range(1, len(stock_prices)):
    current_stock_price = stock_prices[i]
    time_to_expiry = (len(stock_prices) - i) * dt
    new_delta = norm.cdf((np.log(current_stock_price/k) + (r + 0.5*sigma**2)*time_to_expiry) / (sigma * np.sqrt(time_to_expiry)))
    hedge_position += (new_delta - delta) * call_prices[i]
    hedges.append(hedge_position)
    delta = new_delta
    final_profit += call_prices[i] - (delta * current_stock_price + hedge_position)
  return final_profit, hedges