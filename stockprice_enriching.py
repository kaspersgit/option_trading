#%%
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
# %%
def cherry_pick(df, OutOfMoney = 1.1, minDTE =3, maxDTE = 10, minVolOIrate = 5, type = 'calls'):
    calls = (df['symbolType'] == 'Call') & (OutOfMoney * df['baseLastPrice'] < df['strikePrice']) & (df['daysToExpiration'] <= maxDTE) & (df['daysToExpiration'] >= minDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    puts = ( OutOfMoney * df['baseLastPrice'] > df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    selected_df = df[calls]
    selected_df.reset_index(drop=True, inplace=True)
    return(selected_df)

def get_mature_df(df, symbolType=['Call']):
    # Select only mature cases (and exclude options with less then 5 days to expiration)
    df_select = df[df['expirationDate'] < datetime.today().strftime('%Y-%m-%d')]
    df_select = df_select[df_select['daysToExpiration'] > 4]
    df_select = df_select[df_select['symbolType'].isin(symbolType)]
    return(df_select)
#%%
# Load and clean data
df = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-26.csv')
df2 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-29.csv')
df = df.merge(df2)
cols = ['volume','openInterest']
df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
df[cols] = df[cols].apply(lambda x: x.astype('int'))
#%%
i = 1
ticker = cherry_df['baseSymbol'][i]
start_date = cherry_df['current_date'][i]
end_date = cherry_df['expirationDate'][i]
stock = yf.Ticker(ticker)
stock_to_ed = stock.history(start=start_date, end=end_date)

# %%
tickers_list = ["SPY","AAPL"]
tickers = ','.join(tickers_list)
data = yf.download(tickers, start="2017-01-01", end="2017-04-30",
                   group_by="ticker")
data.loc['2017-01-03']['SPY']['Close']

#%%
expiration_dates = df['expirationDate'].unique()


start_date = df['current_date'][0]
start_date = pd.to_datetime(start_date)
start_date_p1 = (start_date + pd.DateOffset(1)).strftime('%Y-%m-%d')
nextBD = (1 * pd.offsets.BDay() + start_date).strftime('%Y-%m-%d')

final_df = []
for end in expiration_dates:
    if end <= start_date_p1:
        continue
    print('Working with enddate {}'.format(end))
    tickers_list = df[df['expirationDate']==end]['baseSymbol'].unique()
    tickers = ','.join(tickers_list)
    data = yf.download(tickers, start=start_date, end=end)
    
    # next business day opening
    openbd = data.loc[nextBD]['Open']

    # Get max high and min low
    highs = data.loc[start_date_p1::]['High'].max()
    lows = data.loc[start_date_p1::]['Low'].min()
    last_close = data['Close'].tail(1).mean()
  
    if len(tickers_list)==1:
        highs = pd.DataFrame({'baseSymbol': [tickers], 'maxPrice': [highs]})
        lows = pd.DataFrame({'baseSymbol': [tickers], 'minPrice': [lows]})
        openbd = pd.DataFrame({'baseSymbol': [tickers], 'nextBDopen': [openbd]})
        last_close = pd.DataFrame({'baseSymbol': [tickers], 'lastClose': [last_close]})
    else:
        highs = highs.reset_index()
        highs.columns=['baseSymbol','maxPrice']
        lows = lows.reset_index()
        lows.columns=['baseSymbol','minPrice']
        openbd = openbd.reset_index()
        openbd.columns=['baseSymbol','nextBDopen']
        last_close = last_close.reset_index()
        last_close.columns=['baseSymbol','lastClose']

    #temp_df = pd.merge(temp_df, highs, how='left', on='baseSymbol')
    temp_df = pd.merge(highs, lows, how='left', on=['baseSymbol'])
    temp_df = pd.merge(temp_df, openbd, how='left', on=['baseSymbol'])
    temp_df = pd.merge(temp_df, last_close, how='left', on=['baseSymbol'])
    temp_df['expirationDate'] = end
    if len(final_df) == 0:
        final_df = temp_df
    else:
        final_df = final_df.append(temp_df)
final_df.reset_index(drop=True, inplace=True)
# %%
# Merge into big df
df_enr = pd.merge(df,final_df, how='left', on=['baseSymbol','expirationDate'])
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr['nextBDopen'] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr['nextBDopen'] * 0.9 >= df_enr['minPrice'],1,0)
# Too be safe, if reached both targets, put 110% target to zero
df_enr['high_plus10p'] = np.where(df_enr['low_min10p'] == 1,0,df_enr['high_plus10p'])

# %%
cherry_df = cherry_pick(df_enr, OutOfMoney = 1.1, minDTE = 5, maxDTE = 10, minVolOIrate = 1.9)
cherry_df.describe()

#%%%
# Get profit 
df_enr['revenue'] = np.where(df_enr['high_plus10p'] == 1, 1.1*df_enr['nextBDopen'], df_enr['lastClose'])
df_enr['revenue'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr['nextBDopen'], df_enr['revenue'])
df_enr['profit'] = df_enr['revenue'] - df_enr['nextBDopen']
#%%
# Select only mature cases (and exclude options with less then 5 days to expiration)
df_mature_call = get_mature_df(df_enr)
df_mature_call.describe()

#%%
# Predicting
# All included regression 
df_mature_call_copy = df_mature_call.copy()
used_cols = ['baseLastPrice', 'strikePrice', 'daysToExpiration',
       'midpoint', 'lastPrice',
       'volumeOpenInterestRatio' ]
train_set = df_mature_call_copy.sample(frac=0.75, random_state=0)
test_set = df_mature_call_copy.drop(train_set.index)

X = train_set[used_cols]
Y = train_set['high_plus10p']
X = sm.add_constant(X)


# Fit and summarize OLS model
mod = sm.Logit(Y, X)

res = mod.fit(maxiter=100)
print(res.summary())

pred = res.predict(test_set[used_cols])
test_set['prediction'] = pred
#%%
### check single stock
yf.ticker('')