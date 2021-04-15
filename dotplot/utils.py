import numpy as np
import pandas as pd

DEFAULT_CLUSTERPROFILE_KEYS = {
    'item_key': 'Description', 'group_key': 'group', 'sizes_key': 'Count', 'color_key': 'pvalue',
    'circle_key': 'qvalue'
}


def merge_clusterprofile_results(dataframes, groups, group_key='group', term_list=None):
    assert len(dataframes) == len(groups)
    merged_df = None
    for _dataframe, _group in zip(dataframes, groups):
        if term_list is None:
            _sub_df = _dataframe
        else:
            _sub_df = _dataframe[_dataframe.index.isin(term_list)]
        if not _sub_df.empty:
            # _sub_df[group_key] = _group # SettingWithCopyWarning
            _sub_df.insert(_sub_df.shape[1], group_key, _group)
            if merged_df is not None:
                merged_df = pd.concat((merged_df, _sub_df))
            else:
                merged_df = _sub_df
    merged_df['qvalue'] = merged_df['qvalue'].map(lambda x: -np.log10(x))
    merged_df['pvalue'] = merged_df['pvalue'].map(lambda x: -np.log10(x))
    merged_df['p.adjust'] = merged_df['p.adjust'].map(lambda x: -np.log10(x))
    return merged_df
