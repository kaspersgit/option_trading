# Load packages
import pandas as pd

# Helper functions
def cherry_pick(df, OutOfMoney = 1.1, maxDTE = 10, minVolOIrate = 5):
    calls = (df['symbolType'] == 'Call') & (OutOfMoney * df['baseLastPrice'] < df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    putts = ( OutOfMoney * df['baseLastPrice'] > df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    selected_df = df[calls]
    return(selected_df)

# import data
df = pd.read_csv('barchart_unusual_activity_2020-06-24.csv')

# apply filters
cherry_df = cherry_pick(df, OutOfMoney = 1.1, maxDTE = 14, minVolOIrate = 3)
selected_options_df = cherry_df[['baseSymbol','baseLastPrice','symbolType','strikePrice','expirationDate','lastPrice', 'volume', 'openInterest',
       'volumeOpenInterestRatio','current_date']]
selected_options_df.to_csv('options_high_volumeOIratio_20200624.csv')




