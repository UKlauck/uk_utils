import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
#from numbers import Integral
from sklearn.preprocessing import LabelEncoder


from scipy.stats import multivariate_normal

def get_colormap():
    return plt.cm.tab10


# -----------------------------------------------------------

def plot_decision_regions(X=None, y=None,
                          classifier=None,
                          data=None, features=None, classes=None,  # for DataFrames only
                          x1range=None, x2range=None,
                          x1=0, x2=1,
                          gridpoints=100,
                          # figsize=(6, 6),
                          plot_legend=True,
                          ax=None
                          ):
    '''
    '''
    # Select colormaps
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    if X is None and data is None and (x1range is None or x2range is None):
        print('ERROR: Cannot plot decision regions without specifying data or a range')
        return

    cmap1 = get_colormap()  # for the mesh
    cmap2 = get_colormap()  # for scatter plots

    # Collect input features from a pandas DataFrame
    if data is not None:
        if type(data) is not DataFrame:
            print('ERROR: Parameter data must be of type DataFrame')
            return

        # Features
        X = data[features].values
        feat_names = features

        # Class / target output
        y = data[classes].values
    else:
        feat_names = (f'$x_{x1+1}$', f'$x_{x2+1}$')

    # Use a label encoder. Does not change the encoding for int's.
    if y is None:
        y = np.zeros(X.shape[0], dtype=np.int8)

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(y)
    class_names = class_encoder.classes_

    # Compute bounds/intervals for the color normalizer used by pcolormesh and scatter
    bounds = np.linspace(-0.5, len(class_names) - 0.5, len(class_names) + 1)

    # Set plot range for x- and y-direction
    if x1range is not None:
        x1_min, x1_max = x1range[0], x1range[1] + \
                         (x1range[1] - x1range[0]) / gridpoints
    else:
        dx = (X[:, x1].max() - X[:, x1].min()) / 20
        x1_min, x1_max = X[:, x1].min() - dx, X[:, x1].max() + dx

    if x2range is not None:
        x2_min, x2_max = x2range[0], x2range[1] + \
                         (x2range[1] - x2range[0]) / gridpoints
    else:
        dy = (X[:, x2].max() - X[:, x2].min()) / 20
        x2_min, x2_max = X[:, x2].min() - dy, X[:, x2].max() + dy

    if ax is None:
        ax = plt.gca()

    # Plot the class
    if classifier is not None:
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min) / gridpoints),
                               np.arange(x2_min, x2_max, (x2_max - x2_min) / gridpoints))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        if Z.ndim > 1:
            if Z.shape[1]>1:
                Z = np.argmax(Z, axis=1)
            else:
                Z = Z.round().ravel()

        Z = class_encoder.transform(Z)
        Z = Z.reshape(xx1.shape).astype(np.int32)  # .round()

        plt.grid(False)
        plt.pcolormesh(xx1, xx2, Z,
                       norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                       alpha=0.3,
                       cmap=cmap1)

        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    else:
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

    # Plot training points
    if X is not None:
        if y is not None:
            scatter = ax.scatter(X[:, x1], X[:, x2], c=y, edgecolors='k', linewidth=0.5,
                                 norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                                 cmap=cmap2)
        else:
            scatter = ax.scatter(X[:, x1], X[:, x2], edgecolors='k', linewidth=0.5,
                                 norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                                 cmap=cmap2)

    if len(class_names) > 1 and plot_legend:
        legend1 = ax.legend(*(scatter.legend_elements()[0], class_names),
                            loc='best', title='Classes')
        ax.add_artist(legend1)

    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])


# -----------------------------------------------------------

def plot_nv_1d(mu=0.0, var=1.0, xlim=(-5, 5), nx=100):
    x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / nx)
    y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-1.0 / 2.0 * np.square(x - mu) / var)
    plt.plot(x, y)


# -----------------------------------------------------------

def plot_nv_3D(mu, Sigma, N=50, xlim=None, ylim=None, zlim=None, zticks=None, figsize=(5, 5)):
    if xlim is not None:
        X = np.linspace(xlim[0], xlim[1], N)
    else:
        X = np.linspace(-5, 5, N)

    if ylim is not None:
        Y = np.linspace(ylim[0], ylim[1], N)
    else:
        Y = np.linspace(-5, 5, N)

    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma, True)
    Z = F.pdf(pos)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True,
                    cmap=cm.inferno)

    cset = ax.contour(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.inferno)

    # Adjust the limits, ticks and view angle
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(xlim)

    ax.set_zlim(-0.15, Z.max())
    # ax.set_zticks(np.linspace(0, Z.max(), 5))
    ax.view_init(27, 300)


# -----------------------------------------------------------

def plot_nv_contour(mu, Sigma, N=50, xlim=None, ylim=None):
    if xlim is not None:
        X = np.linspace(xlim[0], xlim[1], N)
    else:
        X = np.linspace(-5, 5, N)

    if ylim is not None:
        Y = np.linspace(ylim[0], ylim[1], N)
    else:
        Y = np.linspace(-5, 5, N)

    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma, True)
    Z = F.pdf(pos)

    cset = plt.contour(X, Y, Z, cmap=cm.inferno)

    # Adjust the limits, ticks and view angle
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(xlim)


# -----------------------------------------------------------
def plot_history(hist,
                 figsize=(8, 5),
                 ylim=(0, None),
                 combined_plot=True,
                 loss_only=False,
                 acc_only=False,
                 metrics=None,
                 title=None,
                 filename=None):
    """
    Plots the loss and/or accuracy values returned by a call to fit()

    Parameters
    ----------
    hist: the hist object returned by the fit() function

    figsize: list or tuple with 2 values
        size for the figure
    ylim: list or tuple with 2 values
        Min and max value for the y axis
    combined_plot: True: plot all curves in a combined plot
                   False: use separate plots
    loss_only: True: Only loss will be plotted
    acc_only: True: Only accuracy will be plotted
    metrics: None. str or tuple/list of str. Names of the metrics to plot
    title: title of the plot
    filename: filename for saving the plot

    Returns
    -------
    ---
    """

    legend = []

    # Check parameter metrics
    if metrics is not None:
        if type(metrics) == str:
            metrics = [metrics]

        if type(metrics[0]) is not str:
            print('ERROR: Invalid value in metrics')
            return

    # Make sure, that a y axis is labeled for plots with just one curve
    if len(hist.history.keys()) == 1: combined_plot = False

    if combined_plot: plt.figure(figsize=figsize)

    for m in hist.history.keys():  # hist.params['metrics']:

        if loss_only and 'loss' not in m: continue
        if acc_only and 'acc' not in m: continue

        if metrics is not None:
            metric_found = False
            for m_ in metrics:
                if m_ in m:
                    metric_found = True
                    break
            if not metric_found:
                continue

        if not combined_plot: plt.figure(figsize=figsize)

        y = hist.history[m]
        legend.append(m)

        # x_ticks = np.linspace(1, len(y), len(y))
        x_ticks = np.array(hist.epoch) + 1

        plt.plot(x_ticks, y)

        if not combined_plot:
            plt.xlabel('epoch')
            if len(y) <= 10:
                plt.xticks(x_ticks)

            plt.ylabel(m)
            plt.ylim(ylim)
            if title is not None:
                plt.title(title)
            if filename is not None:
                plt.savefig(filename)
            plt.show()

    if combined_plot:
        try:
            plt.xlabel('epoch')
            if len(y) <= 10:
                plt.xticks(x_ticks)
            plt.ylabel('')
            plt.ylim(ylim)
            plt.legend(legend)
            if title is not None:
                plt.title(title)
            if filename is not None:
                plt.savefig(filename)
            plt.show()
        except:
            pass


# -----------------------------------------------------------
def plot_accuracy(hist, figsize=(8, 5), ylim=(0, 1), title=None):
    plot_history(hist, figsize, ylim, title=title, acc_only=True)


# -----------------------------------------------------------
def plot_loss(hist, figsize=(8, 5), ylim=(0, None), title=None):
    plot_history(hist, figsize, ylim, title=title, loss_only=True)


# -----------------------------------------------------------
def plot_silhouette(X, model, filename=None):
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_clusters = model.n_clusters

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 5)  # fig.set_size_inches(18, 7)

    # Left subplot is the silhouette plot
    ax1.set_xlim([None, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Perform clustering
    cluster_labels = model.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10

    cmap = get_colormap()

    min_s = 0

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        min_s = min(min_s, ith_cluster_silhouette_values.min())

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap.colors[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.9,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Plot a vertical line marking the average silhouette score
    ax1.axvline(x=silhouette_avg, color="black", linestyle="--")

    # Set x-axis ticks
    x_ticks = []
    for tick in (-1, -0.8, -0.6, -0.4, -0.2):
        if min_s < tick + 0.15:
            x_ticks.append(tick)
    ax1.set_xticks(x_ticks + [0, 0.2, 0.4, 0.6, 0.8, 1])

    if min_s > -0.1:
        ax1.set_xlim(-0.1, 1)

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    plot_decision_regions(X, cluster_labels, plot_legend=False)

    # Mark cluster centers with white circles ...
    centers = model.cluster_centers_

    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    # ... and plot cluster indexes
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_aspect(1)

    plt.suptitle(
        f"Silhouette plot for {n_clusters} clusters",
        fontsize=14,
        fontweight="bold",
    )

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.show()