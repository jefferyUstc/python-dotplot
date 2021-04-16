import fastcluster
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import numpy as np

def fast_cluster(array, method, metric):
    euclidean_methods = ('centroid', 'median', 'ward')
    euclidean = metric == 'euclidean' and method in \
            euclidean_methods   
    if euclidean or self.method == 'single':
        linkage = fastcluster.linkage_vector(array,
                                              method=method,
                                              metric=metric)
    else:
        linkage = fastcluster.linkage(array, method=method,
                                          metric=metric)
    return linkage


def make_linkage(array, method, metric):
    try :
        return fast_cluster(array, method, metric)
    except ImportError:
        if len(array) >= 10000:
                msg = ("Clustering large matrix with scipy. Installing "
                       "`fastcluster` may give better performance.")
                warnings.warn(msg)
        
        linkage = linkage(array, method=method,
                                    metric=metric)
        return linkage
    
    
def cluster_hierarchy(data, method, axis, metric='euclidean', n_clusters=None, no_plot=True):
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
    if isinstance(data, pd.DataFrame):
        array = data.values
    else:
        array = np.asarray(data)
        data = pd.DataFrame(array)
    
    linkage = make_linkage(array, method, metric)
    
    result = dendrogram(linkage, no_plot=no_plot)
    
    if n_clusters != None:
        cut_result = cluster.hierarchy.cut_tree(linkage, n_clusters=n_clusters)
        df_cut = pd.DataFrame(cut_result.flatten())
        label = df_cut.iloc[:,0].sort_values(ascending=True, inplace=False).index.values
        return data.index.values[label]
    
    if axis == 0:
        need = data.index.values[result['leaves']]
    elif axis == 1:
        need = data.columns.values[result['leaves']]
        
    return need
if __name__ == "__main__":
    test_df = pd.DataFrame([[11,3],[1,2],[25,1],[5,0]], index=["label_1","label_2","label_3","label_4"])
    result = cluster_hierarchy(test_df, 'median', 0)
    print(result)
