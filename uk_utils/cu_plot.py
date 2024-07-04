from matplotlib import pyplot as plt
import seaborn as sns

import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

save_it = False

# ------------------------------------------------------------------------------
def save_graph(sg=False):
    global save_it
    save_it = sg


# ------------------------------------------------------------------------------
def show_plot(file_name=None):
    global save_it
    if save_it and file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

    plt.show()


# ------------------------------------------------------------------------------
def show_graph(file_name=None):
    show_plot(file_name)


# ------------------------------------------------------------------------------
def plot_time_series(signal,
                     figsize=(10, 2.5),
                     show_ts=True,
                     show_acf=False, show_pacf=False, lags=50, alpha=1,
                     show_qq=False, show_prob=False, show_hist=False,
                     title=None):
    rows = 0
    cols = 2
    if show_ts:
        rows += 1
    if show_acf or show_pacf:
        rows += 1
    if show_qq or show_prob:
        rows += 1

    layout = (rows, cols)

    tsax = None
    acfax = None
    pacfax = None
    qqax = None
    probax = None

    _ = plt.figure(figsize=(figsize[0], figsize[1] * rows))

    row = 0
    if show_ts:
        tsax = plt.subplot2grid(layout, (row, 0), colspan=2)
        row += 1

    if show_acf and show_pacf:
        acfax = plt.subplot2grid(layout, (row, 0), colspan=1)
        pacfax = plt.subplot2grid(layout, (row, 1), colspan=1)
        row += 1
    elif show_acf:
        acfax = plt.subplot2grid(layout, (row, 0), colspan=1)
        row += 1
    elif show_pacf:
        pacfax = plt.subplot2grid(layout, (row, 0), colspan=1)
        row += 1

    if show_qq and (show_prob or show_hist):
        qqax = plt.subplot2grid(layout, (row, 0), colspan=1)
        probax = plt.subplot2grid(layout, (row, 1), colspan=1)
        row += 1
    elif show_qq:
        qqax = plt.subplot2grid(layout, (row, 0), colspan=1)
        row += 1
    elif show_prob or show_hist:
        probax = plt.subplot2grid(layout, (row, 0), colspan=1)
        row += 1

    if tsax:
        tsax.plot(signal)
        if title is not None:
            tsax.set_title(title)
        tsax.set_ylabel('Time Series')

    if acfax:
        smt.graphics.plot_acf(signal, lags=lags, ax=acfax, alpha=alpha)
        acfax.set_ylabel('ACF')
        acfax.set_title('Autocorrelation')

    if pacfax:
        smt.graphics.plot_pacf(signal, lags=lags, ax=pacfax, alpha=alpha)
        pacfax.set_ylabel('PACF')
        pacfax.set_title('Partial Autocorrelation')

    if qqax:
        sm.qqplot(signal, line='s', ax=qqax)
        # acfax.set_ylabel('Q-Q')
        qqax.set_title('Quantiles Plot')

    if probax:
        if show_hist:
            sns.histplot(signal, ax=probax)
            probax.set_title('Amplitude Histogram')
        else:
            stats.probplot(signal, sparams=(signal.mean(), signal.std()), plot=probax)
        # pacfax.set_ylabel('PACF')
        # pacfax.set_title(None)

    plt.tight_layout()


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
