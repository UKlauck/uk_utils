import numpy as np
import math

from numpy.linalg import det

def generate_1D_normal(mean, var, n=10, cl=None, seed=None):
    """Generates samples from a 1D normal distribution

    Parameters
    ----------
    mean: scalar
        Mean of the distribution
    var: scalar
        Variance of the distribution
    n:  int
        number of samples to generate
    cl: int
        if given, the class number of the samples generated
    seed: int
        seed for random number generator


    Returns
    -------
    x: array of shape (n)
        The generated samples
    y: array of shape (n)
        Contains the class numbers of the samples (only if cl is given)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.normal(mean, var, n)

    if cl is not None:
        y = np.ndarray(x.shape[0], dtype=np.int8)
        y[:] = cl
        return x, y
    else:
        return x


def generate_nD_normal(mean, corr, var, n=10, cl=None, seed=None):
    """Generates samples from a d-dimensional normal distribution

    Parameters
    ----------
    mean: array of shape (d)
        Mean vector of the d variables/features
    corr: array of shape (d, d)
        Correlation matrix of the variables/features
    var: array of shape (d)
        Variances of the d variables/features
    n:  int
        number of samples to generate
    cl: int
        if given, the class number of the samples generated

    Returns
    -------
    x: array of shape (n,d)
        The generated samples
    y: array of shape (n)
        Contains the class numbers of the samples (only if cl is given)
    """

    mean = np.asarray(mean)
    corr = np.asarray(corr)
    var = np.asarray(var)

    cov = np.empty_like(corr)

    var2 = np.sqrt(var.reshape(-1, 1))
    cov = corr * np.multiply(var2, var2.T)

    if seed is not None:
        np.random.seed(seed)

    x = np.random.multivariate_normal(mean, cov, n, 'ignore')

    if cl is not None:
        y = np.ndarray(x.shape[0], dtype=np.int8)
        y[:] = cl
        return x, y
    else:
        return x


def create_1D_linfunction(n=10, m=1, c=0, noise=0.5, seed=None):
    """Generates values from a linear function y=m*x+c with additional normally distributed noise

    Parameters
    ----------
    n: int
        number of function values to generate
    m: float
        slope of the function
    c: float
        intercept of the function
    noise: float
        number of samples to generate

    Returns
    -------
    x: array of shape (n)
        x values (unordered). 0 <= xi <= n
    y: array of shape (n)
        y values
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.random(n)*10
    y = m*x+c + np.random.randn(n)*noise
    return x, y


def create_sequence(ts, n_inp=1, n_outp=1, method='timesteps'):
    """Creates a sequence suitable for input of a neural network for time sereis analysis

    Parameters
    ----------
    ts: array of shape (n)
        time series data
    n_inp: int
        number of input neurons
    n_outp: int
        number of output neurons
    method: str
        unused a the moment

    Returns
    -------
    X: array of shape (n, n_inp)
        x values
    Y: array of shape (n, m_outp)
        y values
    """

    X, Y = list(), list()
    ts = np.asarray(ts)

    N = ts.shape[0] - n_outp - n_inp + 1

    for i in range(N):
        X.append(ts[i:i+n_inp])
        Y.append(ts[i+n_inp:i+n_inp+n_outp])

    X = np.array(X).reshape((N, n_inp))
    Y = np.array(Y).reshape((N, n_outp))

    return np.array(X), np.array(Y)


def create_multivariate_sequence(ts, window_size=1, predictions=1, delay=0, reshape_1D=True):
    """Creates a sequence suitable for input of a neural network for time series analysis

    Parameters
    ----------
    ts: array of shape (n)
        time series data
    window_size: int
        number of inputs into model
    predictions: int
        number of predictions expected from model
    delay: int
        number of predictions expected from model
    reshape_1D: boolean
        wether or not to collapse 1D data

    Returns
    -------
    X: array of shape (n, window_size, n_channels or 0)
        x values
    Y: array of shape (n, predictions, n_channels or 0)
        y values
    """

    X, Y = list(), list()
    ts = np.asarray(ts)

    if ts.ndim == 1:
        n_channels = 1
        ts = ts.reshape(-1, 1)
    else:
        n_channels = ts.shape[1]

    N = ts.shape[0] - predictions - window_size - delay + 1

    for i in range(N):
        X.append(ts[i:i + window_size, :])
        Y.append(ts[i + window_size + delay:i + window_size + delay + predictions, :])

    if n_channels == 1 and reshape_1D:
        X = np.asarray(X).reshape((N, window_size))
        Y = np.asarray(Y).reshape((N, predictions))
    else:
        X = np.asarray(X).reshape((N, window_size, n_channels))
        Y = np.asarray(Y).reshape((N, predictions, n_channels))

    return X, Y
