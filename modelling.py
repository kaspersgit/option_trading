#%%
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import pickle
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

# Logistic regressions model
def logitModel(test_set, train_set, ex_vars, target):
    X = train_set[ex_vars]
    Y = train_set[target]
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
    return(test_set, res)

# Logistic regressions model
def RandomForest(test_set, train_set, ex_vars, target):
    X = train_set[ex_vars]
    Y = train_set[target]

    # Fit and summarize OLS model
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X,Y)

    # sometimes seem to need to add the constant
    Xtest = test_set[ex_vars]
    pred = clf.predict_proba(Xtest)
    test_set['prediction'] = pred[:,1]
    return(test_set, clf)



def up_for_trade(df):
    # In the money based on the last base price
    df=df[df['inTheMoney']!=1]
    # In the money based on the nextBDopen 
    df=df[(~df['nextBDopen'].isnull()) & (df['strikePrice']>df['nextBDopen'])]
    # Stock price lower than 500 $
    df=df[df['baseLastPrice'] < 500]
    # Return result
    return(df)

def enrich_df(df):
    df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
    df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
    df['inTheMoney'] = np.where((df['symbolType']=='Call') & (df['baseLastPrice'] >= df['strikePrice']),1,0)
    df['inTheMoney'] = np.where((df['symbolType']=='Putt') & (df['baseLastPrice'] <= df['strikePrice']),1,df['inTheMoney'])
    df['nrOptions'] = 1
    df['strikePriceCum'] = df['strikePrice']

    df.sort_values(['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ], inplace=True)

    df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','inTheMoney','volume','openInterest'
            ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'
            ]).agg({'baseSymbol':'count', 'strikePrice':'mean', 'volume':'sum', 'openInterest':'sum'
            }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
            }).reset_index()
    
    # only give info about calls with higher strike price
    df_option_inv_cum = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','strikePriceCum','inTheMoney','volume','openInterest','nrOptions'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney','strikePrice'
        ]).sum().sort_values('strikePrice', ascending = False
        ).groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney']
        ).agg({'volume':'cumsum', 'openInterest':'cumsum', 'nrOptions':'cumsum', 'strikePriceCum':'cumsum'
        }).rename(columns={'volume':'volumeCumSum', 'openInterest':'openInterestCumSum', 'nrOptions':'nrHigherOptions', 'strikePriceCum':'higherStrikePriceCum'
        }).reset_index()
    
    df_call = df_symbol[df_symbol['symbolType']=='Call']
    df_call.rename(columns={'nrOccurences':'nrCalls', 'meanStrikePrice':'meanStrikeCall', 'volume':'volumeCall', 'openInterest':'openInterestCall'}, inplace=True)
    df_call.drop(columns=['symbolType'], inplace=True)
    df_put = df_symbol[df_symbol['symbolType']=='Put']
    df_put.rename(columns={'nrOccurences':'nrPuts', 'meanStrikePrice':'meanStrikePut', 'volume':'volumePut', 'openInterest':'openInterestPut'}, inplace=True)
    df_put.drop(columns=['symbolType'], inplace=True)

    # Add summarized data from Calls and Puts to df
    df = pd.merge(df,df_call, how='left', on=['exportedAt','baseSymbol','expirationDate','inTheMoney'])
    df = pd.merge(df,df_put, how='left', on=['exportedAt','baseSymbol','expirationDate','inTheMoney'])
    df = pd.merge(df,df_option_inv_cum, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney','strikePrice'])
    # set nr of occurences to 0 when NaN
    df[['nrCalls','nrPuts','volumeCall','volumePut']] = df[['nrCalls','nrPuts','volumeCall','volumePut']].fillna(0)

    df['meanStrikeCallPerc'] = df['meanStrikeCall'] / df['baseLastPrice']
    df['meanStrikePutPerc'] = df['meanStrikePut'] / df['baseLastPrice']
    df['midpointPerc'] = df['midpoint'] / df['baseLastPrice']
    df['meanHigherStrike'] = df['higherStrikePriceCum'] / df['nrHigherOptions']
    return(df)
#%%
# Load and clean data
df20200624 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-24.csv')
df20200625 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-25.csv')
df20200629 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-29.csv')
df20200701 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-01.csv')
df20200703 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-03.csv')
df20200706 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-06.csv')
df20200708 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-08.csv')
df20200709 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-09.csv')
df20200710 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-10.csv')

df = pd.concat([df20200624,df20200625,df20200629,df20200701,df20200703,df20200706,df20200708],ignore_index=True)

cols = ['volume','openInterest']
df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
df[cols] = df[cols].apply(lambda x: x.str.replace('%',''))
df[cols] = df[cols].apply(lambda x: x.astype('float'))

# Newer so above is already applied in scraping script
df = pd.concat([df,df20200709,df20200710],ignore_index=True)
df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')
cols = ['volatility']
df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
df[cols] = df[cols].apply(lambda x: x.str.replace('%',''))
df[cols] = df[cols].apply(lambda x: x.astype('float'))

#%%
# Adding some additional columns
df = enrich_df(df)
#%%
# reducing size of dataset- only select interesting options
df = get_mature_df(df)
df = df[df['inTheMoney']==0]
# Adding the stockprices
final_df = add_stock_price(df)

# %%
# Merge into big df
df_enr = pd.merge(df,final_df, how='left', on=['baseSymbol','expirationDate','exportedAt'])

# Add target variables
# Add flag if stock raised 50%
df_enr['high_plus50p'] = np.where(df_enr['nextBDopen'] * 1.5 <= df_enr['maxPrice'],1,0)
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr['nextBDopen'] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr['nextBDopen'] * 0.9 >= df_enr['minPrice'],1,0)
# Too be safe, if reached both targets, put 110% target to zero
df_enr['high_plus10p'] = np.where(df_enr['low_min10p'] == 1,0,df_enr['high_plus10p'])
# Add flag if they reached strikeprice
df_enr['reachedStrike'] = np.where(df_enr['maxPrice'] >= df_enr['strikePrice'],1,0)
# Add flag if reached 110% of strikeprice
df_enr['reachedStrike110p'] = np.where(df_enr['maxPrice'] >= 1.1*df_enr['strikePrice'],1,0)

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
df_enr['revenueStrike110p'] = np.where(df_enr['reachedStrike110p'] == 1, 1.1 * df_enr['strikePrice'], df_enr['lastClose'])
df_enr['profitStrike110p'] = df_enr['revenueStrike110p'] - df_enr['nextBDopen']
df_enr['profitStrikePerc110p'] = df_enr['profitStrike110p']/df_enr['nextBDopen']
df_enr['units'] = 1000/df_enr['nextBDopen']
df_enr['weightedProfitStrike110p'] = df_enr['units'] * df_enr['profitStrike110p']
#%%
# Select only mature cases (and exclude options with less then 5 days to expiration)
df_mature_call = get_mature_df(df_enr)
df_mature_call.describe()

# TODO aggregate options of the same company 
# now extra weight is on characteristics of e.g. TESLA 
# as that stock rose and it had a lot of options showing activity

#%%
# Predicting
# Only include data points in regression we would act on in real life
df_regr = up_for_trade(df_mature_call)

# variable to be predicted
target = 'reachedStrike110p'
# input used to predict with
ex_vars = ['baseLastPrice',
       #'strikePrice',
       #'volOIrate',
        'daysToExpiration',
        'lastPrice',
        #'volume',
        'openInterest',
       'volumeOpenInterestRatio',
        'volatility',
       #'priceDiff',
        'priceDiffPerc',
        'nrCalls',
        #'meanStrikeCall',
        'volumeCall',
        #'openInterestCall',
        'nrPuts',
        #'volumeCumSum', 
        #'openInterestCumSum',
       #'nrHigherOptions', 
       #'higherStrikePriceCum',
        #'meanStrikeCallPerc',
        #'meanHigherStrike',
        'midpointPerc',
       #'nextBDopen'
       ]

#train_set = df_regr.sample(frac=0.85, random_state=1)
train_set = df_regr[df_regr['exportedAt']<'2020-07-08']
test_set = df_regr.drop(train_set.index).reset_index(drop=True)

# Training logistic regression
logit_preds, logit_model = logitModel(test_set, train_set, ex_vars, target=target)
rf_preds, rf_model = RandomForest(test_set, train_set, ex_vars, target=target)

# mean profitability of cases with prediction > 50%
# Assuming we would invest a certain amount equally spread among the stocks
predicted_df = logit_preds.copy()
threshold = 0.2
filtered_df = predicted_df[(predicted_df['prediction']>threshold) 
    #& (predicted_df['nextBDopen'] < 0.95*predicted_df['strikePrice'])
    ][['baseSymbol','exportedAt','expirationDate','strikePrice','nextBDopen','profitStrike','profitStrike110p']]
filtered_df['units'] = 10000/filtered_df['nextBDopen']
filtered_df['weightedProfit'] = filtered_df['profitStrike110p'] * filtered_df['units']
filtered_mean = filtered_df.mean()
print('Profit margin: {} \nActing on {} call options'.format(round(filtered_mean['weightedProfit'] / 10000,4),len(filtered_df)))

# Showing the ROC and AUC
from sklearn import metrics
auc = metrics.roc_auc_score(predicted_df[target], predicted_df['prediction'])
print("The Area under the ROC is {}".format(round(auc,3)))\
# to visualize
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(predicted_df[target], predicted_df['prediction'])
plt.plot(fpr, tpr)
# %%

#%%
threshold = 0.7
# to filter out already into the money options
filtered_set = logit_preds[(logit_preds['inTheMoney']==0) 
    & (logit_preds['priceDiffPerc'] > 1.05) 
    #& (logit_preds['strikePrice'] > logit_preds['nextBDopen'])
    #& (logit_preds['strikePrice'] < 1000)
 ]
filtered_set[filtered_set['prediction']>threshold].mean()

# %%
filtered_set[filtered_set['prediction']>threshold].describe()

# %%
data = yf.download('CLDR', start='2020-06-23', end='2020-07-01')
data
# %%
# IF happy save models
# LOGIT
logit_model.save('/Users/kasper.de-harder/gits/option_trading/modelLogit')

#%%
# RandomForest
# save the model to disk
filename = '/Users/kasper.de-harder/gits/option_trading/RandomForest.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# %%
