import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta, MO

def plotHistogram(serie, show_highest=0.99):
    # Only show highest 99%
    serie_ = serie.copy()
    serie_ = serie_.sort_values().reset_index(drop=True)

    l_serie = len(serie_)
    serie_ = serie_[serie_.index <= l_serie * show_highest]
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=serie_, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

def plotMultipleLines(df, xcol = 'week_start', ycol = 'reachedStrikePrice', groupcol = 'strikePriceIncreaseBin'):
    df_grouped = df[[xcol,groupcol,ycol]].groupby([xcol,groupcol]).agg({ycol:['sum', 'count']})
    df_grouped.reset_index(inplace=True)
    df_grouped['percentageSuccess'] = df_grouped[ycol]['sum'] / df_grouped[ycol]['count']


    df_pivot = df_grouped.pivot(index=xcol, columns=groupcol, values='percentageSuccess')

    fig, ax = plt.subplots()

    df_pivot.plot(ax=ax)

    ax.set_title('Share of options reaching strike by extraction date')

def AddWeekStart(df, col = 'exportedAt'):
    df_ = df.copy()

    # Change 'myday' to contains dates as datetime objects
    df_[col] = pd.to_datetime(df_[col])
    # 'daysoffset' will container the weekday, as integers
    df_['daysoffset'] = df_[col].apply(lambda x: x.weekday())
    # We apply, row by row (axis=1) a timedelta operation
    df_['week_start'] = df_.apply(lambda x: x[col] - datetime.timedelta(days=x['daysoffset']), axis=1)

    return(df_)

def BinColumn(df, col='strikePriceIncrease'):
    df_ = df.copy()

    # taken from https://pbpython.com/pandas-qcut-cut.html
    df_[col+'Bin'] = pd.cut(df_[col], bins=np.linspace(1.05, 1.30, 6))

    return(df_)

if __name__ == '__main__':
    df['strikePriceIncrease'] = df['strikePrice'] / df['baseLastPrice']
    df = BinColumn(df)
    df = AddWeekStart(df)
    plotMultipleLines(df, xcol = 'week_start', ycol = 'reachedStrikePrice', groupcol = 'strikePriceIncreaseBin')
