import AFE_var
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def backend_testing(parameter_function, returns, time):
    portfolioAtDayN = 100000 # Suppose we invest 100.000$ on day n
    days_estimate = 252
    testing_period = 252
    alpha1 = 0.05 # "We are (1-alpha)% sure that the loss will not exceed .."
    alpha2 = 0.01 # "We are (1-alpha)% sure that the loss will not exceed .."

    VaR_95_1 = np.zeros(testing_period)
    ES_95_1 = np.zeros(testing_period)
    VaR_99_1 = np.zeros(testing_period)
    ES_99_1 = np.zeros(testing_period)
    exceedance_95 = np.zeros(testing_period) 
    exceedance_99 = np.zeros(testing_period) 
    loss = np.zeros(testing_period)

    for i in range(0,testing_period,1):
        mean_return = 0 # Assumption
        used_returns = returns.values[-(testing_period-i+days_estimate):-(testing_period-i)]
        VaR_95_1[i], ES_95_1[i] = parameter_function(time, returns, alpha=alpha1)[0],parameter_function(time, returns, alpha=alpha1)[1]
        VaR_99_1[i], ES_99_1[i] = parameter_function(time, returns, alpha=alpha2)[0], parameter_function(time, returns, alpha=alpha2)[1]

        loss[i] = -portfolioAtDayN*returns.values[-(testing_period-i)]/100
        exceedance_95[i] = VaR_95_1[i] < loss[i]
        exceedance_99[i] = VaR_99_1[i] < loss[i]

    return [VaR_95_1, ES_95_1, exceedance_95], [VaR_99_1, ES_99_1, exceedance_99], loss

def plot_backend_testing(parameter_function, returns, time, testing_period):
    vee_95_1, vee_99_1, loss = backend_testing(parameter_function, returns, time)
    VaR_95_1, ES_95_1, exceedance_95 = vee_95_1[0], vee_95_1[1], vee_95_1[2]
    VaR_99_1, ES_99_1, exceedance_99 = vee_99_1[0], vee_99_1[1], vee_99_1[2]
    plt.figure(figsize=(12,6), dpi=200)
    plt.plot(returns.index[-(testing_period+1):-1], VaR_95_1, c = mcolors.TABLEAU_COLORS['tab:blue'], label = "Value-at-Risk-95% (1 day ahead)")
    plt.plot(returns.index[-(testing_period+1):-1], VaR_99_1, c = mcolors.TABLEAU_COLORS['tab:olive'], label = "Value-at-Risk-99% (1 day ahead)")
    plt.plot(returns.index[-(testing_period+1):-1], loss, c = mcolors.TABLEAU_COLORS['tab:red'], label = "Loss")
    plt.scatter(returns.index[-(testing_period+1):-1][exceedance_95 == 1], VaR_95_1[exceedance_95 == 1], s = 100*1.5, marker="x", color="black", label="Exceedances")
    plt.scatter(returns.index[-(testing_period+1):-1][exceedance_99 == 1], VaR_99_1[exceedance_99 == 1], s = 100*1.5, marker="x", color="black")
    plt.xlabel("Dates", fontsize = 12)
    plt.ylabel("Losses [$]", fontsize = 12)
    plt.title("Backtesting of Market VaR with Normal Distribution", fontsize = 14, y=1.03)
    plt.legend()
    plt.show()

def coverage_test(exceedances, testing_period, alpha):
    N = testing_period

    pi_exp_alpha = alpha
    pi_obs_alpha = np.sum(exceedances == 1)/N
    n1_alpha = np.sum(exceedances == 1)
    n0_alpha = np.sum(exceedances == 0)

    LR_uc_VaR = -2*np.log(((pi_exp_alpha**n1_alpha)*(1-pi_exp_alpha)**n0_alpha)/((pi_obs_alpha**n1_alpha)*(1-pi_obs_alpha)**n0_alpha))
    print("The corresponding likelihood ratio statistic for VaR is: {}".format((LR_uc_VaR)))
    return LR_uc_VaR

def cluster_test(exceedances, testing_period):
    N = testing_period

    T_01 = 0 # violation after non-violation
    T_00 = 0 # non-violation after non-violation
    T_11 = 0 # violation after violation
    T_10 = 0 # non-violation after violation

    for i in range(1, N):
        if exceedances[i]:
            if exceedances[i - 1]:
                T_11 += 1 
            else:
                T_01 += 1 
        else:
            if exceedances[i - 1]:
                T_10 += 1
            else:
                T_00 += 1

    T_1 = exceedances[1:].sum()
    T_0 = N-1 - T_1
    pi_01 = T_01/(T_00 + T_01)
    pi_11 = T_11/(T_10 + T_11)
    pi_obs = T_1/(N-1)

    LR_ind_VaR = -2*np.log(((pi_obs**T_1)*(1-pi_obs)**T_0)/((pi_01**T_01*(1-pi_01)**T_00)*(pi_11**T_11*(1-pi_11)**T_10)))
    print("The corresponding likelihood ratio statistic for VaR is: {}".format((LR_ind_VaR)))
    return LR_ind_VaR




