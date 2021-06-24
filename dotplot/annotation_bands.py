"""
module for draw annotation bands.
PS. This implementation mainly refers to the way of seaborn.clustermap
"""

import itertools
from typing import Union, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams["font.sans-serif"] = "Arial"


def _process_colors(colors: Union[pd.DataFrame, pd.Series],
                    index_order: Union[Sequence[Union[str, int]], None] = None):
    """
    borrowed from seaborn
    """
    if not isinstance(colors, (pd.DataFrame, pd.Series)):
        raise TypeError('`colors` should be pandas.DataFrame or pandas.Series')
    if index_order is not None:
        colors = colors.reindex(index_order)
    colors = colors.astype(object).fillna('white')  # TODO, to alpha
    if isinstance(colors, pd.DataFrame):
        labels = list(colors.columns)
        colors = colors.T.values
    else:
        labels = [''] if colors.name is None else [colors.name]
        colors = colors.values
    try:
        to_rgb(colors[0])
        colors = list(map(to_rgb, colors))
    except ValueError:
        colors = [list(map(to_rgb, item)) for item in colors]
    return colors, labels


def _color_list_to_matrix_and_cmap(colors, axis=0):
    """
    borrowed from seaborn
    """
    if any(issubclass(type(item), list) for item in colors):
        all_colors = set(itertools.chain(*colors))
        n = len(colors)  # number of fields
        m = len(colors[0])  # number of observations
    else:
        all_colors = set(colors)
        n = 1
        m = len(colors)
        colors = [colors]
    color_to_value = dict((col, i) for i, col in enumerate(all_colors))

    matrix = np.array([color_to_value[c]
                       for color in colors for c in color])

    shape = (n, m)
    matrix = matrix.reshape(shape)
    if axis == 0:
        # row-side:
        matrix = matrix.T

    cmap = mpl.colors.ListedColormap(all_colors)
    return matrix, cmap


def _determine_ticks(labels: Sequence[str]):
    """
    center the tick positions
    """
    num = len(labels)
    return [item + .5 for item in range(num)]


def draw_heatmap(colors: Union[pd.DataFrame, pd.Series],
                 axes: mpl.axes.Axes, axis=0,
                 index_order: Union[Sequence[Union[str, int]], None] = None, **kwargs):
    """

    :param colors: pd.Series or pd.DataFrame
    :param axes: matplotlib axes object
    :param axis: 0 or 1
    :param index_order: the order of colors
    :param kwargs: passed to `axes.pcolormesh`
    :return:
    """
    colors = colors.copy()
    # if index_order is not None:
    #     index_order = list(index_order)[::-1]
    colors, labels = _process_colors(colors, index_order)
    matrix, cmap = _color_list_to_matrix_and_cmap(colors, axis=axis)
    axes.pcolormesh(matrix, cmap=cmap, **kwargs)
    if axis == 1:
        axes.set_yticks(_determine_ticks(labels))
        axes.set_yticklabels(labels)
        _ = axes.set_xticks([])
    elif axis == 0:
        axes.set_xticks(_determine_ticks(labels))
        axes.set_xticklabels(labels, rotation=90)
        _ = axes.set_yticks([])
    else:
        raise ValueError('axis must be 0 or 1.')
    for item in ['left', 'right', 'bottom', 'top']:
        axes.spines[item].set_visible(False)
