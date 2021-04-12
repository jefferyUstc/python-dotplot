import matplotlib as mpl


def get_colormap(color_list: list, segment=1000):
    return mpl.colors.LinearSegmentedColormap.from_list('color_list', color_list, N=segment)
