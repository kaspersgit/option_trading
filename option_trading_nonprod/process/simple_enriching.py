import pandas as pd
import numpy as np

def simpleEnriching(df):
    """
    Adding some basic columns to dataframe regarding option details
    """
    df_ = df.copy()

    # progress towards strikeprice
    df_['strikePriceProgress'] = (df_['maxPrice'] - df_['baseLastPrice']) / (df_['strikePrice'] - df_['baseLastPrice'])
    df_['strikePriceProgressCapped'] = np.where(df_['strikePriceProgress'] >= 1, 1.0, df_['strikePriceProgress'])
    # Add columns
    df_['strikePricePerc'] = df_['strikePrice'] / df_['baseLastPrice']
    # expected profit
    df_['expPercIncrease'] = df_['strikePricePerc'] * df_['prob']
    # profitability (percentage increase from stock price to max price)
    df_['maxProfitability'] = df_['maxPrice'] / df_['baseLastPrice']

    # buy one of each stock of a certain amount worth of stocks (df_['baseLastPrice'] vs 100)
    # below we implement 100 dollars worth of each stock
    df_['stocksBought'] = 100 / df_['baseLastPrice']
    df_['cost'] = df_['stocksBought'] * df_['baseLastPrice']
    df_['revenue'] = df_['stocksBought'] * np.where(df_['reachedStrikePrice'] == 1, df_['strikePrice'], df_['finalPrice'])
    df_['profit'] = df_['revenue'] - df_['cost']
    df_['profitPerc'] = df_['profit'] / df_['cost']

    # bin the strike price increase
    # Bucket strike price increase
    bins = [-np.inf, 1.1, 1.15, 1.20, 1.25, 1.3, 1.4, 1.5, np.inf]
    labels = ['5%-10%', '10%-15%', '15%-20%', '20%-25%', '25%-30%', '30%-40%', '40%-50%', '>50%']

    df_['strikePricePercBin'] = pd.cut(df_['strikePricePerc'], bins=bins, labels=labels)

    set_difference = set(df_.keys()) - set(df.keys())
    col_difference = list(set_difference)

    print(f'Added {len(col_difference)} columns')
    print(f'Added columns: {col_difference}')

    return df_

def addTargets(df):
    df_ = df.copy()
    df_['reachedStrikePrice'] = np.where(df_['maxPrice'] >= df_['strikePrice'], 1, 0)
    df_['percStrikeReached'] = (df_['maxPrice'] - df_['baseLastPrice']) / (
            df_['strikePrice'] - df_['baseLastPrice'])
    df_['finalPriceHigher'] = np.where(df_['finalPrice'] >= df_['baseLastPrice'], 1, 0)
    return df_

def cleanDF(df):
    df_ = df.copy()
    df_ = df_.drop_duplicates(subset=['baseSymbol', 'symbolType', 'strikePrice', 'expirationDate', 'exportedAt'])
    # TODO belows seems to filter out everything almost?
    # df_ = df_[~df_['midpoint'].isna()]
    return df_