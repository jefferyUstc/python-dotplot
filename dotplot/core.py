import math
from os import PathLike
from typing import Union, Sequence, Callable, Dict

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams["font.sans-serif"] = "Arial"


class DotPlot(object):
    DEFAULT_ITEM_HEIGHT = 0.3
    DEFAULT_ITEM_WIDTH = 0.3
    DEFAULT_LEGENDS_WIDTH = .45
    MIN_FIGURE_HEIGHT = 3
    DEFAULT_BAND_ITEM_LENGTH = DEFAULT_ITEM_HEIGHT

    # TODO implement annotation band
    def __init__(self, df_size: pd.DataFrame,
                 df_color: Union[pd.DataFrame, None] = None,
                 df_circle: Union[pd.DataFrame, None] = None,
                 df_annotation: Union[pd.DataFrame, None] = None,
                 ):
        """
        Construction a `DotPlot` object from `df_size` and `df_color`

        :param df_size: the DataFrame object represents the scatter size in dotplot
        :param df_color: the DataFrame object represents the color in dotplot
        """
        __slots__ = ['size_data', 'resized_size_data',
                     'color_data', 'height_item', 'width_item',
                     'circle_data', 'resized_circle_data', 'annotation_data'
                     ]
        if df_color is not None and df_size.shape != df_color.shape:
            raise ValueError('df_size and df_color should have the same dimension')
        if df_circle is not None and df_size.shape != df_circle.shape:
            raise ValueError('df_size and df_circle should have the same dimension')
        if df_annotation is not None and df_size.shape != df_annotation.shape:
            raise ValueError('df_size and df_annotation should have the same row number')

        self.size_data = df_size
        self.color_data = df_color
        self.circle_data = df_circle
        self.height_item, self.width_item = df_size.shape
        self.annotation_data = df_annotation
        self.resized_size_data: Union[pd.DataFrame, None] = None
        self.resized_circle_data: Union[pd.DataFrame, None] = None

    def __get_figure(self):
        _text_max = math.ceil(self.size_data.index.map(len).max() / 15)
        mainplot_height = self.height_item * self.DEFAULT_ITEM_HEIGHT
        mainplot_width = (
                (_text_max + self.width_item) * self.DEFAULT_ITEM_WIDTH
        )
        figure_height = max([self.MIN_FIGURE_HEIGHT, mainplot_height])
        figure_width = mainplot_width + self.DEFAULT_LEGENDS_WIDTH
        if self.annotation_data is not None:
            # figure_width = figure_width + self.DEFAULT_BAND_ITEM_LENGTH * self.annotation_data.shape[1]
            ...
        plt.style.use('seaborn-white')
        fig = plt.figure(figsize=(figure_width, figure_height))
        gs = gridspec.GridSpec(nrows=3, ncols=2, wspace=0.15, hspace=0.15,
                               width_ratios=[mainplot_width, self.DEFAULT_LEGENDS_WIDTH])
        ax = fig.add_subplot(gs[:, 0])
        ax_cbar = fig.add_subplot(gs[2, 1])
        ax_sizes = fig.add_subplot(gs[0, 1])
        ax_circles = fig.add_subplot(gs[1, 1])
        if self.color_data is None:
            ax_cbar.axis('off')
        ax_circles.axis('off')
        return ax, ax_cbar, ax_sizes, ax_circles, fig

    @classmethod
    def parse_from_tidy_data(cls, data_frame: pd.DataFrame, item_key: str, group_key: str, sizes_key: str,
                             color_key: Union[None, str] = None, circle_key: Union[None, str] = None,
                             selected_item: Union[None, Sequence] = None,
                             selected_group: Union[None, Sequence] = None, *,
                             sizes_func: Union[None, Callable] = None, color_func: Union[None, Callable] = None
                             ):
        """

        class method for conveniently constructing DotPlot from tidy data

        :param data_frame:
        :param item_key:
        :param group_key:
        :param sizes_key:
        :param color_key:
        :param selected_item: default None, if specified, this should be subsets of `item_key` in `data_frame`
                              alternatively, this param can be used as self-defined item order definition.
        :param selected_group: Same as `selected_item`, for group order and subset groups
        :param sizes_func:
        :param color_func:
        :param circle_key:
        :return:
        """
        keys = [v for v in [item_key, group_key, sizes_key, color_key, circle_key] if v is not None]
        data_frame = data_frame[keys]
        _original_item_order = data_frame[item_key].tolist()
        _original_item_order = _original_item_order[::-1]
        _original_item_order = sorted(set(_original_item_order), key=_original_item_order.index)
        if sizes_func is not None:
            data_frame[sizes_key] = data_frame[sizes_key].map(sizes_func)
        if color_func is not None:
            data_frame[color_key] = data_frame[color_key].map(color_func)
        keys.remove(item_key)
        keys.remove(group_key)
        data_frame = data_frame.pivot(index=item_key, columns=group_key, values=keys)
        data_frame = data_frame.loc[_original_item_order, :]
        if selected_item is not None:
            data_frame = data_frame.loc[selected_item, :]
        if selected_group is not None:
            data_frame = data_frame.loc[:, selected_group]
        data_frame.columns = data_frame.columns.map(lambda x: '_'.join(x))
        data_frame = data_frame.fillna(0)

        sizes_df, color_df, circle_df = None, None, None
        sizes_df = data_frame.loc[:, data_frame.columns.str.startswith(sizes_key)]
        if color_key is not None:
            color_df = data_frame.loc[:, data_frame.columns.str.startswith(color_key)]
        if circle_key is not None:
            circle_df = data_frame.loc[:, data_frame.columns.str.startswith(circle_key)]
        return cls(sizes_df, color_df, circle_df)

    def __get_coordinates(self, size_factor):
        X = list(range(1, self.width_item + 1)) * self.height_item
        Y = sorted(list(range(1, self.height_item + 1)) * self.width_item)
        return X, Y

    def __draw_dotplot(self, ax, size_factor, cmap, vmin, vmax, **kws):
        dot_color = kws.get('dot_color', '#58000C')
        circle_color = kws.get('circle_color', '#000000')
        kws = kws.copy()
        for _value in ['dot_title', 'circle_title', 'colorbar_title', 'dot_color', 'circle_color']:
            _ = kws.pop(_value, None)

        X, Y = self.__get_coordinates(size_factor)
        if self.color_data is None:
            sct = ax.scatter(X, Y, c=dot_color, s=self.resized_size_data.values.flatten(),
                             edgecolors='none', linewidths=0, vmin=vmin, vmax=vmax, cmap=cmap, **kws)
        else:
            sct = ax.scatter(X, Y, c=self.color_data.values.flatten(), s=self.resized_size_data.values.flatten(),
                             edgecolors='none', linewidths=0, vmin=vmin, vmax=vmax, cmap=cmap, **kws)
        sct_circle = None
        if self.circle_data is not None:
            sct_circle = ax.scatter(X, Y, c='none', s=self.resized_circle_data.values.flatten(),
                                    edgecolors=circle_color, marker='o', vmin=vmin, vmax=vmax, linestyle='--')
        width, height = self.width_item, self.height_item
        ax.set_xlim([0.5, width + 0.5])
        ax.set_ylim([0.6, height + 0.6])
        ax.set_xticks(range(1, width + 1))
        ax.set_yticks(range(1, height + 1))
        ax.set_xticklabels(self.size_data.columns.tolist(), rotation='vertical')
        ax.set_yticklabels(self.size_data.index.tolist())
        ax.tick_params(axis='y', length=5, labelsize=15, direction='out')
        ax.tick_params(axis='x', length=5, labelsize=15, direction='out')
        return sct, sct_circle

    @staticmethod
    def __draw_color_bar(ax, sct: mpl.collections.PathCollection, cmap, vmin, vmax, ylabel):
        gradient = np.linspace(1, 0, 500)
        gradient = gradient[:, np.newaxis]
        _ = ax.imshow(gradient, aspect='auto', cmap=cmap, origin='upper', extent=[.2, 0.3, 0.5, -0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax_cbar2 = ax.twinx()
        _ = ax_cbar2.set_yticks([0, 1000])
        if vmax is None:
            vmax = math.ceil(sct.get_array().max())
        if vmin is None:
            vmin = math.floor(sct.get_array().min())
        _ = ax_cbar2.set_yticklabels([vmin, vmax])
        _ = ax_cbar2.set_ylabel(ylabel)

    @staticmethod
    def __draw_legend(ax, sct: mpl.collections.PathCollection, size_factor, title, circle=False, color=None):
        handles, labels = sct.legend_elements(prop="sizes", alpha=1,
                                              func=lambda x: x / size_factor,
                                              color=color
                                              )
        if len(handles) > 3:
            handles = np.asarray(handles)
            labels = np.asarray(labels)
            handles = handles[[0, math.ceil(len(handles) / 2), -1]]
            labels = labels[[0, math.ceil(len(labels) / 2), -1]]
        if circle:
            from matplotlib.lines import Line2D
            for i, _item in enumerate(handles):
                xdata, ydata = _item.get_data()
                marker_size = _item.get_markersize()
                handles[i] = Line2D(xdata, ydata, color='white', marker='$\u25CC$',
                                    markeredgecolor=color, markersize=marker_size)
        _ = ax.legend(handles, labels, title=title, loc='center left')  # bbox_to_anchor=(0.9, 0., 0.4, 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def __preprocess_data(self, size_factor, cluster_row=False, cluster_col=False, **kwargs):

        method = kwargs.get('cluster_method', 'ward')
        metric = kwargs.get('cluster_metric', 'euclidean')
        n_clusters = kwargs.get('cluster_n', None)

        if cluster_row or cluster_col:
            from .hierarchical import cluster_hierarchy
            if cluster_row:
                _index = cluster_hierarchy(self.size_data, axis=0, method=method,
                                           metric=metric, n_clusters=n_clusters)
            else:
                _index = cluster_hierarchy(self.size_data, axis=1, method=method,
                                           metric=metric, n_clusters=n_clusters)
            obj_data = self.__dict__.copy()
            for _obj_attr, _obj in obj_data.items():
                if not _obj_attr.startswith('__'):
                    if isinstance(_obj, pd.DataFrame):
                        if cluster_row:
                            _obj = _obj.loc[_index, :]
                        if cluster_col:
                            _obj = _obj.loc[:, _index]
                        setattr(self, _obj_attr, _obj)
        self.resized_size_data = self.size_data.applymap(func=lambda x: x * size_factor)
        if self.circle_data is not None:
            self.resized_circle_data = self.circle_data.applymap(func=lambda x: x * size_factor)

    def plot(self, size_factor: float = 15,
             vmin: float = 0, vmax: float = None,
             path: Union[PathLike, None] = None,
             cmap: Union[str, mpl.colors.Colormap] = 'Reds',
             cluster_row: bool = False, cluster_col: bool = False,
             cluster_kws: Union[Dict, None] = None, **kwargs
             ):
        """

        :param size_factor: `size factor` * `value` for the actually representation of scatter size in the final figure
        :param vmin: `vmin` in `matplotlib.pyplot.scatter`
        :param vmax: `vmax` in `matplotlib.pyplot.scatter`
        :param path: path to save the figure
        :param cmap: color map supported by matplotlib
        :param kwargs: dot_title, circle_title, colorbar_title, dot_color, circle_color
                    other kwargs are passed to `matplotlib.Axes.scatter`
        :param cluster_row, whether to cluster the row
        :param cluster_col, whether to cluster the row
        :param cluster_kws, key args for cluster, including `cluster_method`, `cluster_metric`， 'cluster_n'
        :return:
        """
        self.__preprocess_data(size_factor, cluster_row=cluster_row, cluster_col=cluster_col,
                               **cluster_kws if cluster_kws is not None else {}
                               )
        ax, ax_cbar, ax_sizes, ax_circles, fig = self.__get_figure()
        scatter, sct_circle = self.__draw_dotplot(ax, size_factor, cmap, vmin, vmax)
        self.__draw_legend(ax_sizes, scatter, size_factor,
                           color=kwargs.get('dot_color', '#58000C'),  # dot legend color
                           title=kwargs.get('dot_title', 'Sizes'))
        if sct_circle is not None:
            self.__draw_legend(ax_circles, sct_circle, size_factor,
                               color=kwargs.get('circle_color', '#000000'),
                               title=kwargs.get('circle_title', 'Circles'),
                               circle=True)
        if self.color_data is not None:
            self.__draw_color_bar(ax_cbar, scatter, cmap, vmin, vmax,
                                  ylabel=kwargs.get('colorbar_title', '-log10(pvalue)'))
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight')  #
        return scatter

    def __str__(self):
        return 'DotPlot object with data point in shape %s' % str(self.size_data.shape)

    __repr__ = __str__
