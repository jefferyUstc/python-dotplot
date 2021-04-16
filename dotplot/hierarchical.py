import warnings

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree


def fast_cluster(array, method, metric):
    import fastcluster
    euclidean_methods = ('centroid', 'median', 'ward')
    euclidean = metric == 'euclidean' and method in euclidean_methods
    if euclidean or method == 'single':
        _linkage = fastcluster.linkage_vector(array,
                                              method=method,
                                              metric=metric)
    else:
        _linkage = fastcluster.linkage(array, method=method,
                                       metric=metric)
    return _linkage


def make_linkage(array, method, metric):
    try:
        return fast_cluster(array, method, metric)
    except ImportError:
        if len(array) >= 10000:
            msg = ("Clustering large matrix with scipy. Installing "
                   "`fastcluster` may give better performance.")
            warnings.warn(msg)

        _linkage = linkage(array, method=method, metric=metric)
        return _linkage


def cluster_hierarchy(data, method, axis, metric='euclidean', n_clusters=None):
    """
    data :pandas.DataFrame
        Rectangular data
    method :str, 'single', 'centroid', 'median', 'ward'
    axis : int, optional
        Which axis to use to calculate linkage. 0 is rows, 1 is columns.
    metric : "eulidean"
    n_cluster: int, optional
        return the cut tree.
    no_plot: bool, optional
        When True, the final rendering is not performed. This is
        useful if only the data structures computed for the rendering
        are needed or if matplotlib is not available.
    """
    data = data.copy()
    if axis == 1:
        data = data.T
    array = data.values
    _linkage = make_linkage(array, method, metric)

    if n_clusters is not None:
        cut_result = cut_tree(_linkage, n_clusters=n_clusters)
        df_cut = pd.DataFrame(cut_result.flatten())
        label = df_cut.iloc[:, 0].sort_values(ascending=True, inplace=False).index.values
        return data.index.values[label]
    _result = dendrogram(_linkage, no_plot=True)
    _reordered_index = data.index.values[_result['leaves']]
    return _reordered_index
