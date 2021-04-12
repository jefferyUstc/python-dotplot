import numpy as np
import pandas as pd


def merge_clusterprofile_results(dataframes, groups, term_list=None):
    assert len(dataframes) == len(groups)
    merged_df = None
    for _dataframe, _group in zip(dataframes, groups):
        if term_list is None:
            _sub_df = _dataframe
        else:
            _sub_df = _dataframe[_dataframe.index.isin(term_list)]
        if not _sub_df.empty:
            _sub_df['group'] = _group
            if merged_df is not None:
                merged_df = pd.concat((merged_df, _sub_df))
            else:
                merged_df = _sub_df
    merged_df = merged_df[['Description', 'pvalue', 'GeneRatio', 'group']]
    merged_df['GeneRatio'] = merged_df.GeneRatio.map(lambda x: int(x.split('/')[0]))
    merged_df['pvalue'] = merged_df['pvalue'].map(lambda x: -np.log10(x))
    return merged_df
