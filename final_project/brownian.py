import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def log_likelihood(theta, prices, dt):
    """
    Compute the log likelihood of a Geometric Brownian Motion.

    Parameters:
    ----------
    theta : numpy.ndarray
        An array of shape (2,) containing the parameters mu and sigma of the GBM.
    prices : numpy.ndarray
        An array of shape (n,) containing the observed prices.
    dt : float
        The time step between prices.

    Returns:
    -------
    float
        The log likelihood of the GBM.
    """

    n = len(prices)

    # Extract the parameters.
    mu = theta[0]
    sigma = theta[1]

    # Compute the log likelihood.
    ll = -n/2*np.log(2*np.pi*dt) - 1/(2*dt)*np.sum((np.log(prices[1:]/prices[:-1]) - (mu - 0.5*sigma**2)*dt)**2)
    return ll

def estimate_gbm_parameters(prices, dt):
    """
    Estimate the parameters of a Geometric Brownian Motion.

    Parameters:
    ----------
    prices : numpy.ndarray
        An array of shape (n,) containing the observed prices.
    dt : float
        The time step between prices.

    Returns:
    -------
    tuple
        A tuple containing the estimated parameters (mu, sigma) of the GBM.
    """

    # Set up the optimization problem.
    def neg_log_likelihood(theta):
        return -log_likelihood(theta, prices, dt)

    # Set the initial guess for the parameters.
    theta_init = np.array([0.0, 0.1])

    # Run the optimization.
    result = minimize(neg_log_likelihood, theta_init)

    # Extract the estimated parameters.
    mu, sigma = result.x

    return mu, sigma

def plot_gbm(index, prices, simulated_prices, stock_name):
    # Plot simulated prices
    plt.figure(figsize=(10, 6))
    plt.plot(index, prices, 'b-', alpha=1, label='Actual Price')
    plt.plot(index, simulated_prices.T, 'r-', alpha=0.05)
    plt.plot(index, np.mean(simulated_prices, axis=0), 'green', alpha=1, label='Average Simulated Price')

    # get legend handles and labels
    blue_patch = mpatches.Patch(color='blue', label='Actual Price')
    red_patch = mpatches.Patch(color='red', label='Simulated Prices')
    green_patch = mpatches.Patch(color='green', label='Average Simulated Price')

    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Geometric Brownian Motion - Simulated Stock Prices for ' + stock_name)
    plt.legend(handles=[blue_patch, red_patch, green_patch], labels=['Actual Prices', 'Simulated Prices', 'Average Simulated Price'])
    plt.grid(True)
    