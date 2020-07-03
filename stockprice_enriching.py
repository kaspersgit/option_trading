#%%
import yfinance as yf
import pandas as pd
from datetime import datetime
# %%
def cherry_pick(df, OutOfMoney = 1.1, minDTE =3, maxDTE = 10, minVolOIrate = 5, type = 'calls'):
    calls = (df['symbolType'] == 'Call') & (OutOfMoney * df['baseLastPrice'] < df['strikePrice']) & (df['daysToExpiration'] <= maxDTE) & (df['daysToExpiration'] >= minDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    puts = ( OutOfMoney * df['baseLastPrice'] > df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    selected_df = df[calls]
    selected_df.reset_index(drop=True, inplace=True)
    return(selected_df)
#%%
df = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-26.csv')
cherry_df = cherry_pick(df, OutOfMoney = 1.1, minDTE = 3, maxDTE = 10, minVolOIrate = 3)
selected_options_df = cherry_df[['baseSymbol','baseLastPrice','symbolType','strikePrice','expirationDate','lastPrice', 'volume', 'openInterest',
       'volumeOpenInterestRatio','current_date']]
#%%
i = 1
ticker = cherry_df['baseSymbol'][i]
start_date = cherry_df['current_date'][i]
end_date = cherry_df['expirationDate'][i]
stock = yf.Ticker(ticker)
stock_to_ed = stock.history(start=start_date, end=end_date)

# %%
tickers_list = ["SPY","AAPL"]
data = yf.download(tickers_list, start="2017-01-01", end="2017-04-30",
                   group_by="ticker")
data.loc['2017-01-03']['SPY']['Close']

#%%
expiration_dates = df['expirationDate'].unique()
start_date = df['current_date'][0]

for end in expiration_dates:
    tickers_list = df[df['expirationDate']==end]['baseSymbol'].unique()
    data = yf.download(tickers_list, start=start_date, end=end)
    data.loc['2017-01-03']['SPY']['Close']


# %%
