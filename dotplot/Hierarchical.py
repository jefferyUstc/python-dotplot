from seaborn.matrix import _DendrogramPlotter
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
def cluster_hierarchy(data, method, axis):
    """
    data :pandas.DataFrame
        Rectangular data
    method :str, 'single', 'centroid', 'median', 'ward'
    axis : int, optional
        Which axis to use to calculate linkage. 0 is rows, 1 is columns.
    """
    
    cluster = _DendrogramPlotter(data, method=method, metric='euclidean',linkage=None, axis=axis, label=None, rotate=None)
    result = dendrogram(cluster.linkage)
    plt.close()
    
    if axis == 0:
        need = data.index.values[result['leaves']]
    elif axis == 1:
        need = data.columns.values[result['leaves']]
        
    return need
if __name__ == "__main__":
    test_df = pd.DataFrame([[0,3],[1,2],[2,1],[3,0]], index=["label_1","label_2","label_3","label_4"])
    result = cluster_hierarchy(test_df, 'median', 0)
    print(result)
