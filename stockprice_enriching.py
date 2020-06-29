#%%
import yfinance as yf
import pandas as pd
from datetime import datetime
# %%
df = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-26.csv')

# %%
ticker = cherry_df['baseSymbol'][0]
start_date = cherry_df['current_date'][0]
end_date = cherry_df['expirationDate'][0]
stock = yf.Ticker(ticker)
stock.history(start=start_date, end=end_date)

# %%
