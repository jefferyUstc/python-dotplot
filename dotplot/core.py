import math
from os import PathLike
from typing import Union, Sequence, Callable, Dict, List

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt

from .exceptions import ShapeInconsistencyError

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams["font.sans-serif"] = "Arial"

CMAPS_PRESET = ('Reds', 'Blues', 'Purples', 'Oranges', 'Greens', 'Greys')
COLORS_PRESET = ('r', 'b', '#BA55D3', '#FFA500', 'g', '#C0C0C0')


class DotPlot(object):
    DEFAULT_ITEM_HEIGHT = 0.3
    DEFAULT_ITEM_WIDTH = 0.3
    DEFAULT_LEGENDS_WIDTH = .6
    MIN_FIGURE_HEIGHT = 3.5
    DEFAULT_BAND_ITEM_LENGTH = .2

    def __init__(self, df_size: pd.DataFrame,
                 df_color: Union[pd.DataFrame, None] = None,
                 df_circle: Union[pd.DataFrame, None] = None,
                 row_colors: Union[pd.DataFrame, None] = None,
                 col_colors: Union[pd.DataFrame, None] = None,
                 mask_frames: Union[pd.DataFrame, None] = None
                 ):
        """
        Construction a `DotPlot` object from `df_size` and `df_color`

        :param df_size: the DataFrame object represents the scatter size in dotplot
        :param df_color: the DataFrame object represents the color in dotplot
        """
        __slots__ = ['size_data', 'resized_size_data', 'color_data', 'height_item', 'width_item',
                     'circle_data', 'resized_circle_data', 'row_colors', 'col_colors', 'mask_frames',
                     'figure'
                     ]
        if df_color is not None and df_size.shape != df_color.shape:
            raise ShapeInconsistencyError('df_size and df_color should have the same dimension')
        if df_circle is not None and df_size.shape != df_circle.shape:
            raise ShapeInconsistencyError('df_size and df_circle should have the same dimension')
        if row_colors is not None and df_size.shape[0] != len(row_colors):
            raise ShapeInconsistencyError('row_colors has the wrong shape')
        if col_colors is not None and df_size.shape[1] != len(col_colors):
            raise ShapeInconsistencyError('col_colors has the wrong shape')
        if mask_frames is not None and df_size.shape != mask_frames.shape:
            raise ShapeInconsistencyError('df_size and mask_frames should have the same dimension')

        self.size_data = df_size
        self.color_data = df_color
        self.circle_data = df_circle
        self.height_item, self.width_item = df_size.shape
        self.row_colors = row_colors
        self.col_colors = col_colors
        self.resized_size_data: Union[pd.DataFrame, None] = None
        self.resized_circle_data: Union[pd.DataFrame, None] = None
        self.mask_frames = mask_frames
        self.figure = None
        if (self.row_colors is None) and (self.col_colors is None):
            self.col_colors = pd.DataFrame({'': ['#FFFFFF'] * df_size.shape[1]},
                                           index=df_size.columns.tolist())

    def __get_figure(self):
        """
        Figure layout
        :return:
        """
        _text_max = math.ceil(self.size_data.index.map(len).max() / 15)
        mainplot_height = self.height_item * self.DEFAULT_ITEM_HEIGHT
        mainplot_width = (
                (_text_max + self.width_item) * self.DEFAULT_ITEM_WIDTH
        )
        figure_height = max([self.MIN_FIGURE_HEIGHT, mainplot_height])
        n_group = len(np.unique(self.mask_frames.to_numpy())) if self.mask_frames is not None else 1
        figure_width = mainplot_width + self.DEFAULT_LEGENDS_WIDTH * n_group
        band_width, band_height = 0., 0.
        if self.row_colors is not None:
            band_width = self.DEFAULT_BAND_ITEM_LENGTH * self.row_colors.shape[1]
        if self.col_colors is not None:
            band_height = self.DEFAULT_BAND_ITEM_LENGTH * self.col_colors.shape[1]
        figure_width = figure_width + band_width
        figure_height = figure_height + band_height

        plt.style.use('seaborn-white')
        fig = plt.figure(figsize=(figure_width, figure_height))
        self.figure = fig
        gs = gridspec.GridSpec(nrows=2, ncols=3, wspace=0.05, hspace=0.02,
                               width_ratios=[mainplot_width, band_width, self.DEFAULT_LEGENDS_WIDTH * n_group],
                               height_ratios=[band_height, mainplot_height]
                               )
        ax = fig.add_subplot(gs[1, 0])
        ax_row_bands = fig.add_subplot(gs[1, 1])
        ax_col_bands = fig.add_subplot(gs[0, 0])
        ax_abandon = fig.add_subplot(gs[0, 1])
        legend_gs = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=.1, subplot_spec=gs[1, 2])
        gs_sizes_legend = legend_gs[0, 0]
        gs_cbar_legend = legend_gs[2, 0]
        gs_circles_legend = legend_gs[1, 0]

        if self.col_colors is None:
            ax_col_bands.axis('off')
        if self.row_colors is None:
            ax_row_bands.axis('off')
        ax_abandon.axis('off')
        return ax, gs_cbar_legend, gs_sizes_legend, gs_circles_legend, ax_row_bands, ax_col_bands, fig

    @classmethod
    def parse_from_tidy_data(cls, data_frame: pd.DataFrame, item_key: str, group_key: str, sizes_key: str,
                             color_key: Union[None, str] = None, circle_key: Union[None, str] = None,
                             selected_item: Union[None, Sequence] = None,
                             selected_group: Union[None, Sequence] = None,
                             row_colors: Union[None, pd.Series, pd.DataFrame] = None,
                             col_colors: Union[None, pd.Series, pd.DataFrame] = None,
                             mask_frames: Union[None, pd.DataFrame, Sequence[Union[str, int]]] = None,
                             *, sizes_func: Union[None, Callable] = None, color_func: Union[None, Callable] = None
                             ):
        """

        class method for conveniently constructing DotPlot from tidy data

        :param data_frame:
        :param item_key:
        :param group_key:
        :param sizes_key:
        :param color_key:
        :param circle_key:
        :param selected_item: default None, if specified, this should be subsets of `item_key` in `data_frame`
                              alternatively, this param can be used as self-defined item order definition.
        :param selected_group: Same as `selected_item`, for group order and subset groups
        :param col_colors:
        :param row_colors:
        :param mask_frames:
        :param sizes_func: Callable
        :param color_func: Callable
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
        if (mask_frames is not None) and isinstance(mask_frames, Sequence):
            mask_frames = mask_frames if isinstance(mask_frames, List) else list(mask_frames)
            n_row, n_col = sizes_df.shape
            if len(mask_frames) == n_row:
                mask_frames = pd.DataFrame([[item] * n_col for item in mask_frames],
                                           index=sizes_df.index.values, columns=sizes_df.columns.values)
            elif len(mask_frames) == n_col:
                mask_frames = pd.DataFrame([[item] * n_row for item in mask_frames],
                                           index=sizes_df.columns.values, columns=sizes_df.index.values)
                mask_frames = mask_frames.T
            else:
                raise ShapeInconsistencyError('mask frame shape Error.')
        return cls(sizes_df, color_df, circle_df, row_colors=row_colors,
                   col_colors=col_colors, mask_frames=mask_frames)

    def __get_coordinates(self):
        X = list(range(1, self.width_item + 1)) * self.height_item
        Y = sorted(list(range(1, self.height_item + 1)) * self.width_item)
        return X, Y

    def __draw_dotplot(self, ax, cmap, vmin, vmax, size_factor, *, gs_cbar: mpl.gridspec.GridSpec,
                       gs_sizes: mpl.gridspec.GridSpec, gs_circles: mpl.gridspec.GridSpec, **kws):
        X, Y = self.__get_coordinates()
        kws = kws.copy()
        dot_color = kws.pop('dot_color', '#58000C')
        dot_title = kws.pop('dot_title', 'Sizes')
        circle_color = kws.pop('circle_color', '#000000')
        circle_title = kws.pop('circle_title', 'Circles')
        colorbar_title = kws.pop('colorbar_title', '-log10(Pvalue)')

        resized_size_data_array = self.resized_size_data.values.flatten()
        color_data_array_or_str = dot_color if self.color_data is None else self.color_data.values.flatten()
        sct: Union[List[mpl.collections.PathCollection], mpl.collections.PathCollection] = []
        if self.mask_frames is not None:
            masks, n_masks, mask_groups = self.resolve_mask(self.mask_frames)
            if isinstance(color_data_array_or_str, np.ndarray):
                vmax = np.max(self.color_data.values.flatten()) if vmax is None else vmax
                if isinstance(cmap, (str, mpl.colors.Colormap)):
                    cmap = CMAPS_PRESET
                if len(cmap) < n_masks:
                    raise ValueError('too many groups to draw with limited color map')
                for _, (mask, _cmap) in enumerate(zip(masks, cmap)):
                    masked_resized_size_data_array = np.ma.masked_array(resized_size_data_array, mask=mask)
                    _sct = ax.scatter(X, Y, c=color_data_array_or_str, s=masked_resized_size_data_array,
                                      edgecolors='none', linewidths=0, vmin=vmin, vmax=vmax, cmap=_cmap, **kws)
                    sct.append(_sct)
                self.__draw_color_bar(gs_cbar, sct, cmap, vmin, vmax, ylabel=colorbar_title)
                self.__draw_legend(gs_sizes, sct[0], size_factor, color=dot_color, title=dot_title)
            else:
                if isinstance(dot_color, str):
                    dot_color = COLORS_PRESET
                if len(dot_color) < n_masks:
                    raise ValueError('too many groups to draw with limited color')
                for _, (mask, _dot_color) in enumerate(zip(masks, dot_color)):
                    masked_resized_size_data_array = np.ma.masked_array(resized_size_data_array, mask=mask)
                    _sct = ax.scatter(X, Y, c=_dot_color, s=masked_resized_size_data_array,
                                      edgecolors='none', linewidths=0, vmin=vmin, vmax=vmax, **kws)
                    sct.append(_sct)
                self.__draw_legend(gs_sizes, sct, size_factor, color=dot_color, title=mask_groups)
        else:
            if self.color_data is not None:
                vmax = np.max(self.color_data.values.flatten()) if vmax is None else vmax
            sct = ax.scatter(X, Y, c=color_data_array_or_str, s=resized_size_data_array,
                             edgecolors='none', linewidths=0, vmin=vmin, vmax=vmax, cmap=cmap, **kws)
            if self.color_data is not None:
                self.__draw_color_bar(gs_cbar, sct, cmap, vmin, vmax, ylabel=colorbar_title)
            self.__draw_legend(gs_sizes, sct, size_factor, color=dot_color, title=dot_title)
        if self.circle_data is not None:
            sct_circle = ax.scatter(X, Y, c='none', s=self.resized_circle_data.values.flatten(),
                                    edgecolors=circle_color, marker='o', vmin=vmin, vmax=vmax, linestyle='--')
            self.__draw_legend(gs_circles, sct_circle, size_factor, color=circle_color, title=circle_title,
                               circle=True)
        width, height = self.width_item, self.height_item
        ax.set_xlim([0.5, width + 0.5])
        ax.set_ylim([0.6, height + 0.6])
        ax.set_xticks(range(1, width + 1))
        ax.set_yticks(range(1, height + 1))
        ax.set_xticklabels(self.size_data.columns.tolist(), rotation='vertical')
        ax.set_yticklabels(self.size_data.index.tolist())
        ax.tick_params(axis='y', length=5, labelsize=15, direction='out')
        ax.tick_params(axis='x', length=5, labelsize=15, direction='out')

    def __draw_color_bar(self, gs: mpl.gridspec.GridSpec,
                         sct: Union[mpl.collections.PathCollection, Sequence[mpl.collections.PathCollection]],
                         cmap, vmin, vmax, ylabel):
        def _draw_color_bars_core(axes, path_collection: mpl.collections.PathCollection, _cmap, _vmin, _vmax, _ylabel):
            gradient = np.linspace(1, 0, 500)
            gradient = gradient[:, np.newaxis]
            _ = axes.imshow(gradient, aspect='auto', cmap=_cmap, origin='upper')
            axes.set_xticks([])
            axes.set_yticks([])
            ax_cbar2 = axes.twinx()
            if _vmax is None:
                _vmax = math.ceil(path_collection.get_array().max())
            if _vmin is None:
                _vmin = math.floor(path_collection.get_array().min())
            if _ylabel:
                _ = ax_cbar2.set_yticks([0, 1000])
                _ = ax_cbar2.set_yticklabels([_vmin, _vmax])
            else:
                _ = ax_cbar2.set_yticks([])
            _ = ax_cbar2.set_ylabel(_ylabel)

        fig = self.figure
        if isinstance(sct, mpl.collections.PathCollection):
            ax = fig.add_subplot(gs)
            _draw_color_bars_core(ax, sct, cmap, vmin, vmax, ylabel)
        else:
            new_gs = gridspec.GridSpecFromSubplotSpec(1, len(sct), wspace=.2, subplot_spec=gs)
            n_sct = len(sct) - 1
            for i, (_sct, _cmap) in enumerate(zip(sct, cmap)):
                ax = fig.add_subplot(new_gs[0, i])
                _ylabel = ylabel if n_sct == i else ''
                _draw_color_bars_core(ax, _sct, _cmap, vmin, vmax, _ylabel=_ylabel)

    def __draw_legend(self, gs: mpl.gridspec.GridSpec,
                      sct: mpl.collections.PathCollection,
                      size_factor, title, circle=False, color=None):
        def __draw_legend_core(_ax, _sct, _size_factor, _title, _circle=False, _color=None):
            handles, labels = _sct.legend_elements(prop="sizes", alpha=1,
                                                   func=lambda x: x / _size_factor,
                                                   color=_color
                                                   )
            if len(handles) > 3:
                handles = np.asarray(handles)
                labels = np.asarray(labels)
                handles = handles[[0, math.ceil(len(handles) / 2), -1]]
                labels = labels[[0, math.ceil(len(labels) / 2), -1]]
            if _circle:
                from matplotlib.lines import Line2D
                for j, _item in enumerate(handles):
                    xdata, ydata = _item.get_data()
                    marker_size = _item.get_markersize()
                    handles[j] = Line2D(xdata, ydata, color='white', marker='$\u25CC$',
                                        markeredgecolor=_color, markersize=marker_size)
            _ = _ax.legend(handles, labels, title=_title, loc='center left', frameon=False)
            _ax.set_xticks([])
            _ax.set_yticks([])
            for item in ['top', 'bottom', 'left', 'right']:
                _ax.spines[item].set_visible(False)

        fig = self.figure
        if isinstance(sct, mpl.collections.PathCollection):
            ax = fig.add_subplot(gs)
            __draw_legend_core(ax, sct, size_factor, title, circle, color)
        else:
            new_gs = gridspec.GridSpecFromSubplotSpec(1, len(sct), wspace=.5, subplot_spec=gs)
            for i, (_sct, _color, _title) in enumerate(zip(sct, color, title)):
                ax = fig.add_subplot(new_gs[0, i])
                ax.set_facecolor((0, 0, 0, 0))
                __draw_legend_core(ax, _sct, size_factor, _title, circle, _color)

    def __cluster_matrix(self, axis=0, **kwargs):
        from .hierarchical import cluster_hierarchy
        method = kwargs.get('cluster_method', 'ward')
        metric = kwargs.get('cluster_metric', 'euclidean')
        n_clusters = kwargs.get('cluster_n', None)
        _index = cluster_hierarchy(self.size_data, axis=axis, method=method,
                                   metric=metric, n_clusters=n_clusters)
        obj_data = self.__dict__.copy()
        for _obj_attr, _obj in obj_data.items():
            if (not _obj_attr.startswith('__')) and isinstance(_obj, (pd.DataFrame, pd.Series)):
                if _obj_attr in ('row_colors', 'col_colors'):  # TODO may change the action in the future
                    continue
                if axis == 0:
                    _obj = _obj.loc[_index, :]
                elif axis == 1:
                    _obj = _obj.loc[:, _index]
                else:
                    raise ValueError('axis should be 0 or 1.')
                setattr(self, _obj_attr, _obj)

    def __preprocess_data(self, size_factor, cluster_row=False, cluster_col=False, **kwargs):
        if cluster_row or cluster_col:
            if cluster_row:
                self.__cluster_matrix(axis=0, **kwargs)
            if cluster_col:
                self.__cluster_matrix(axis=1, **kwargs)
        self.resized_size_data = self.size_data.applymap(func=lambda x: x * size_factor)
        if self.circle_data is not None:
            self.resized_circle_data = self.circle_data.applymap(func=lambda x: x * size_factor)

    def plot(self, size_factor: float = 15,
             vmin: float = 0, vmax: float = None,
             path: Union[PathLike, None] = None,
             cmap: Union[str, mpl.colors.Colormap,
                         Sequence[Union[str, mpl.colors.Colormap]]] = 'Reds',
             cluster_row: bool = False, cluster_col: bool = False,
             cluster_kws: Union[Dict, None] = None,
             color_band_kws: Union[Dict, None] = None,
             **kwargs
             ):
        """

        :param size_factor: `size factor` * `value` for the actually representation of scatter size in the final figure
        :param vmin: `vmin` in `matplotlib.pyplot.scatter`
        :param vmax: `vmax` in `matplotlib.pyplot.scatter`
        :param path: path to save the figure
        :param cmap: color map supported by matplotlib, can be sequence of cmap when drawing grouped dotplots
        :param cluster_row, whether to cluster the row
        :param cluster_col, whether to cluster the col
        :param cluster_kws, key args for cluster, including `cluster_method`, `cluster_metric`， 'cluster_n'
        :param kwargs: dot_title, circle_title, colorbar_title, dot_color, circle_color
                    other kwargs are passed to `matplotlib.Axes.scatter`. Notably, dot_color can be a
                    color sequence when drawing grouped dotplots
        :param color_band_kws: this kwargs was passed to `matplotlib.axes.Axes.pcolormesh`
        :return:
        """
        self.__preprocess_data(size_factor, cluster_row=cluster_row, cluster_col=cluster_col,
                               **cluster_kws if cluster_kws is not None else {}
                               )
        ax, gs_cbar_legend, gs_sizes_legend, gs_circles_legend, ax_row_bands, ax_col_bands, fig = self.__get_figure()
        self.__draw_dotplot(ax, cmap, vmin, vmax, size_factor, gs_cbar=gs_cbar_legend, gs_sizes=gs_sizes_legend,
                            gs_circles=gs_circles_legend, **kwargs)
        if self.col_colors is not None:
            from .annotation_bands import draw_heatmap
            color_band_kws = {} if color_band_kws is None else color_band_kws
            draw_heatmap(self.col_colors, axes=ax_col_bands,
                         index_order=self.size_data.columns.tolist(), axis=1, **color_band_kws)
        if self.row_colors is not None:
            color_band_kws = {} if color_band_kws is None else color_band_kws
            from .annotation_bands import draw_heatmap
            draw_heatmap(self.row_colors, axes=ax_row_bands,
                         index_order=self.size_data.index.tolist(), axis=0, **color_band_kws)
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight')
        return fig

    @staticmethod
    def resolve_mask(mask_dataframe: pd.DataFrame):
        mask_dataframe = mask_dataframe.applymap(func=lambda x: str(x))
        groups = np.unique(mask_dataframe.to_numpy())
        mappings = dict(zip(groups, [0] * len(groups)))
        group_masks = []
        n_group = 0
        for group in groups:
            if group == 'nan':
                continue
            n_group += 1
            _mappings = mappings.copy()
            _mappings.update({group: 1})
            group_mask = mask_dataframe.applymap(func=lambda x: _mappings[x])
            group_masks.append(group_mask)
        if n_group < 2:
            raise ValueError('group number<2')
        return group_masks, n_group, groups

    def __str__(self):
        return 'DotPlot object with data point in shape %s' % str(self.size_data.shape)

    __repr__ = __str__
