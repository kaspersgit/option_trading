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
    df_select = df[pd.to_datetime(df['expirationDate']) < datetime.today()]
    df_select = df_select[df_select['daysToExpiration'] > 4]
    df_select = df_select[df_select['symbolType'].isin(symbolType)]
    return(df_select)

def add_stock_price(df):
    ## Adding actual stockprices
    # create empty list
    final_df = []
    # Loop through all different scraping date
    start_dates = df['exportedAt'].unique()
    for start_date in start_dates:
        data_df = df[df['exportedAt']==start_date]
        # Get the different dates
        start_date = pd.to_datetime(start_date)
        start_date_p1 = (start_date + pd.DateOffset(1)).strftime('%Y-%m-%d')
        nextBD = (1 * pd.offsets.BDay() + start_date).strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')
        print('Working with data scraped on {}'.format(start_date))

        # Get all different expiration dates
        expiration_dates = data_df['expirationDate'].unique()

        for end in expiration_dates:
            if end <= start_date_p1:
                continue
            print('Working with enddate {}'.format(end))
            tickers_list = data_df[data_df['expirationDate']==end]['baseSymbol'].unique()
            tickers = ','.join(tickers_list)
            data = yf.download(tickers, start=start_date, end=end)
            
            # next business day opening
            # first check if avaialable 
            if nextBD not in data.index:
                continue
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
            temp_df['exportedAt'] = start_date
            if len(final_df) == 0:
                final_df = temp_df
            else:
                final_df = final_df.append(temp_df)
    final_df.reset_index(drop=True, inplace=True)
    return(final_df)
#%%
# Load and clean data
df20200624 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-24.csv')
df20200625 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-25.csv')
df20200626 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-26.csv')
df20200629 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-29.csv')
df20200708 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-08.csv')

df = pd.concat([df20200624,df20200626,df20200629,df20200708],ignore_index=True)
df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')
cols = ['volume','openInterest']
df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
df[cols] = df[cols].apply(lambda x: x.astype('int'))

#%%
# Adding some additional columns
df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
df['inTheMoney'] = np.where(df['baseLastPrice'] >= df['strikePrice'],1,0)
df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate'
        ]).agg({'baseSymbol':'count', 'strikePrice':'mean'
        }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
        }).reset_index()
df = pd.merge(df,df_symbol, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate'])
#%%
# Adding the stockprices
final_df = add_stock_price(df)

# %%
# Merge into big df
df_enr = pd.merge(df,final_df, how='left', on=['baseSymbol','expirationDate','exportedAt'])
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr['nextBDopen'] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr['nextBDopen'] * 0.9 >= df_enr['minPrice'],1,0)
# Too be safe, if reached both targets, put 110% target to zero
df_enr['high_plus10p'] = np.where(df_enr['low_min10p'] == 1,0,df_enr['high_plus10p'])
# Add flag if they reached strikeprice
df_enr['reachedStrike'] = np.where(df_enr['maxPrice'] >= df_enr['strikePrice'],1,0)

# %%
cherry_df = cherry_pick(df_enr, OutOfMoney = 1.1, minDTE = 5, maxDTE = 10, minVolOIrate = 1.9)
cherry_df.describe()

#%%%
# Get profit 
df_enr['revenue'] = np.where(df_enr['high_plus10p'] == 1, 1.1*df_enr['nextBDopen'], df_enr['lastClose'])
df_enr['revenue'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr['nextBDopen'], df_enr['revenue'])
df_enr['profit'] = df_enr['revenue'] - df_enr['nextBDopen']
df_enr['profitPerc'] = df_enr['profit']/df_enr['nextBDopen']


df_enr['revenueStrike'] = np.where(df_enr['reachedStrike'] == 1, df_enr['strikePrice'], df_enr['lastClose'])
df_enr['profitStrike'] = df_enr['revenueStrike'] - df_enr['nextBDopen']
df_enr['profitStrikePerc'] = df_enr['profitStrike']/df_enr['nextBDopen']
#%%
# Select only mature cases (and exclude options with less then 5 days to expiration)
df_mature_call = get_mature_df(df_enr)
df_mature_call.describe()

# TODO aggregate options of the same company 
# now extra weight is on characteristics of e.g. TESLA 
# as that stock rose and it had a lot of options showing activity

#%%
# Predicting
# All included regression 
df_mature_call_copy = df_mature_call.copy()
df_mature_call_copy=df_mature_call_copy[df_mature_call_copy['inTheMoney']!=1]
# variable to be predicted
end_var = 'reachedStrike'
# input used to predict with
ex_vars = [#'baseLastPrice', 
        'priceDiff',
       'daysToExpiration', # not significant as we just have 7 and 8 days
       'midpoint', 
       #'lastPrice', 
       'volumeOpenInterestRatio',
       'nrOccurences',
       'meanStrikePrice',
       #'inTheMoney',
       'priceDiffPerc']

#train_set = df_mature_call_copy.sample(frac=0.85, random_state=1)
train_set = df_mature_call_copy[df_mature_call_copy['exportedAt']!='2020-07-08']
test_set = df_mature_call_copy.drop(train_set.index).reset_index(drop=True)
#train_set=train_set.drop_duplicates(subset=['baseSymbol','expirationDate'])

X = train_set[ex_vars]
Y = train_set[end_var]
X = sm.add_constant(X)


# Fit and summarize OLS model
mod = sm.Logit(Y, X)

res = mod.fit(maxiter=100)
print(res.summary())

# sometimes seem to need to add the constant
Xtest = test_set[ex_vars]
Xtest = sm.add_constant(Xtest, has_constant='add')
pred = res.predict(Xtest)
test_set['prediction'] = pred

from sklearn import metrics
auc = metrics.roc_auc_score(test_set[end_var], test_set['prediction'])
print("The Area under the ROC is {}".format(round(auc,3)))
# Best: 0.766 (high_plus10p all dataponits included)
# to visualize
fpr, tpr, thresholds = metrics.roc_curve(test_set[end_var], test_set['prediction'])
plt.plot(fpr, tpr)
#%%
threshold = 0.5
# to filter out already into the money options
filtered_set = test_set[(test_set['inTheMoney']==0) & (test_set['priceDiffPerc'] > 1.05)]
filtered_set[filtered_set['prediction']>threshold].mean()

# %%
filtered_set[filtered_set['prediction']>threshold].describe()

# %%
data = yf.download('CLDR', start='2020-06-23', end='2020-07-01')
data
# %%
# IF happy save model=
res.save('/Users/kasper.de-harder/gits/option_trading/modelLogit')


# %%
