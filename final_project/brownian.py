import numpy as np
from scipy.optimize import minimize

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