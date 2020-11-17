#%%
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import pickle
# %%
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
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(X,Y)

    # sometimes seem to need to add the constant
    Xtest = test_set[ex_vars]
    pred = clf.predict_proba(Xtest)
    test_set['prediction'] = pred[:,1]
    return(test_set, clf)



def up_for_trade(df):
    # Stock price lower than 500 $
    df=df[df['baseLastPrice'] < 200]
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

def level_enriching(df):
    df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
    df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
    df['inTheMoney'] = np.where((df['symbolType']=='Call') & (df['baseLastPrice'] >= df['strikePrice']),1,0)
    df['inTheMoney'] = np.where((df['symbolType']=='Putt') & (df['baseLastPrice'] <= df['strikePrice']),1,df['inTheMoney'])
    df['nrOptions'] = 1
    df['strikePriceCum'] = df['strikePrice']

    df.sort_values(['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ], inplace=True)

    df_stock = df[['exportedAt','baseSymbol','baseLastPrice']].drop_duplicates()

    df_symbol = df[['exportedAt','baseSymbol','symbolType','daysToExpiration','strikePrice','inTheMoney','volume','openInterest','volatility','lastPrice'
            ]].groupby(['exportedAt','baseSymbol','symbolType','inTheMoney'
            ]).agg({'baseSymbol':'count', 'strikePrice':'mean', 'volume':'sum', 'openInterest':'sum', 'daysToExpiration':'mean', 'lastPrice':'mean', 'volatility':'mean'
            }).rename(columns={'baseSymbol':'nrOptions', 'strikePrice':'meanStrikePrice', 'volume':'sumVolume','openInterest':'sumOpenInterest','daysToExpiration':'meanDaysToExpiration','lastPrice':'meanLastPrice', 'volatility':'meanVolatility'
            }).reset_index()
    
    itm = df['inTheMoney'].unique()
    symbol = df['symbolType'].unique()

    for s in symbol:
        for i in itm:
            if i == 1:
                itm_str = 'Itm'
            elif i == 0:
                itm_str = 'Otm'
            
            base_colname = s + itm_str
            temp_df = df_symbol[(df_symbol['symbolType'] == s) & (df_symbol['inTheMoney'] == i)]

            temp_df['volumeOIratio'] = temp_df['sumVolume'] / temp_df['sumOpenInterest']
            temp_df = temp_df.drop(['inTheMoney'], axis=1)
            temp_df.rename(columns={'nrOccurences':'nr' + base_colname, 'meanStrikePrice':'meanStrike' + base_colname
                , 'sumVolume':'volume' + base_colname, 'sumOpenInterest':'openInterest' + base_colname
                , 'meanDaysToExpiration':'daysToExpiration' + base_colname, 'volumeOIratio':'volumeOIratio' + base_colname
                , 'meanLastPrice':'lastprice' + base_colname, 'meanVolatility':'volatility' + base_colname
                , 'nrOptions':'count' + base_colname}, inplace=True)
            temp_df.drop(columns=['symbolType'], inplace=True)

            df_stock = pd.merge(df_stock, temp_df, how='left', on=['exportedAt','baseSymbol'])

    # set nr of occurences to 0 when NaN
    #df[['nrCalls','nrPuts','volumeCall','volumePut']] = df[['nrCalls','nrPuts','volumeCall','volumePut']].fillna(0)
    return(df_stock)

#%%
# Load and clean data
# import market beat
mb20200624 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-24.csv')
mb20200625 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-25.csv')
mb20200629 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-29.csv')
mb20200701 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-01.csv')
mb20200706 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-06.csv')
mb20200707 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-07.csv')
mb20200709 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-09.csv')
mb20200710 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-10.csv')
mb20200714 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-14.csv')
mb20200715 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-15.csv')
mb20200716 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-16.csv')
mb20200717 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-17.csv')

mb_df = pd.concat([mb20200624,mb20200625,mb20200629,mb20200701,mb20200706,mb20200707
    ,mb20200709,mb20200710,mb20200714,mb20200715,mb20200716,mb20200717],ignore_index=True)
#%%
# reducing size of dataset- only select interesting options
# make fake expiration date to let function work
virt_daysToExpiration = 14
mb_df['expirationDate'] = (pd.to_datetime(mb_df['dataDate']) + timedelta(days=virt_daysToExpiration)).dt.strftime('%Y-%m-%d')
mb_df['exportedAt'] = mb_df['dataDate']
mb_df['baseSymbol'] = mb_df['ticker']
mb_df['daysToExpiration'] = virt_daysToExpiration

#df = df[df['inTheMoney']==0]
# Adding the stockprices
final_df = add_stock_price(mb_df)
df_enr = pd.merge(mb_df,final_df, how='left', on=['baseSymbol','expirationDate','exportedAt'])
#%%
# Add target variables
# What price we think we can buy it for?
# baseLastPrice ?
# nextBDopen ?
buyPrice = 'nextBDopen' 
# Add flag if stock raised 50%
df_enr['high_plus50p'] = np.where(df_enr[buyPrice] * 1.5 <= df_enr['maxPrice'],1,0)
df_enr['high_plus30p'] = np.where(df_enr[buyPrice] * 1.3 <= df_enr['maxPrice'],1,0)
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr[buyPrice] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr[buyPrice] * 0.9 >= df_enr['minPrice'],1,0)
# Too be safe, if reached both targets, put 110% target to zero
df_enr['high_plus10p'] = np.where(df_enr['low_min10p'] == 1,0,df_enr['high_plus10p'])
# Add flag if they reached strikeprice
#df_enr['reachedStrike'] = np.where(df_enr['maxPrice'] >= df_enr['strikePrice'],1,0)
# Add flag if reached 110% of strikeprice
#df_enr['reachedStrike110p'] = np.where(df_enr['maxPrice'] >= 1.1*df_enr['strikePrice'],1,0)

#%%%
# add more columns
#variables
df_enr['indicatorPresent'] = np.where(df_enr['indicators'].isnull(),0,1)
df_enr['upcomingEarning'] = np.where(df_enr['indicators'].str.contains('Upcoming Earnings'),1,0)
df_enr['earningAnnounced'] = np.where(df_enr['indicators'].str.contains('Earnings Announcement'),1,0)
df_enr['callStockVolume'] = df_enr['avgStockVolume'] / df_enr['avgOptionVolume']

# Get profit 
df_enr['revenue'] = np.where(df_enr['high_plus10p'] == 1, 1.1*df_enr[buyPrice], df_enr['lastClose'])
df_enr['revenue'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue'])
df_enr['profit'] = df_enr['revenue'] - df_enr[buyPrice]
df_enr['profitPerc'] = df_enr['profit']/df_enr[buyPrice]

# for high of 30 percent 
df_enr['revenue30p'] = np.where(df_enr['high_plus30p'] == 1, 1.3*df_enr[buyPrice], df_enr['lastClose'])
#f_enr['revenue30p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue30p'])
df_enr['profit30p'] = df_enr['revenue30p'] - df_enr[buyPrice]
df_enr['profitPerc30p'] = df_enr['profit30p']/df_enr[buyPrice]

# for high of 50 percent 
df_enr['revenue50p'] = np.where(df_enr['high_plus50p'] == 1, 1.5*df_enr[buyPrice], df_enr['lastClose'])
#df_enr['revenue50p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue50p'])
df_enr['profit50p'] = df_enr['revenue50p'] - df_enr[buyPrice]
df_enr['profitPerc50p'] = df_enr['profit50p']/df_enr[buyPrice]

# weighted profit
df_enr['units'] = 1000/df_enr[buyPrice]
df_enr['weightedProfit30p'] = df_enr['units'] * df_enr['profit30p']
df_enr['weightedProfit50p'] = df_enr['units'] * df_enr['profit50p']

#%%
# Predicting
# Only include data points in regression we would act on in real life
df_regr = df_enr
# variable to be predicted
target = 'high_plus30p'
# input used to predict with
ex_vars = ['nextBDopen', 
        #'callOptionVolume',
       #'avgOptionVolume',
       'increaseRelative2Avg',
       'indicatorPresent',
       #'upcomingEarning',
       'earningAnnounced',
       #'avgStockVolume',
       ]

#train_set = df_regr.sample(frac=0.85, random_state=1)
train_set = df_regr[df_regr['exportedAt']<'2020-07-12']
test_set = df_regr.drop(train_set.index).reset_index(drop=True)

# Training logistic regression
logit_preds, logit_model = logitModel(test_set, train_set, ex_vars, target=target)
#rf_preds, rf_model = RandomForest(test_set, train_set, ex_vars, target=target)

# mean profitability of cases with prediction > 50%
# Assuming we would invest a certain amount equally spread among the stocks
predicted_df = logit_preds.copy()
threshold = 0.15
filtered_df = predicted_df[(predicted_df['prediction']>threshold) 
    ][['baseSymbol','exportedAt','expirationDate',buyPrice,target,'profit50p','profit30p','profit','lastClose','prediction']]
filtered_df['units'] = 10000/filtered_df[buyPrice]
if target == 'reachedStrik110p':
    profitVar = 'profitStrike110p'
elif target == 'reachedStrike':
    profitVar = 'profitStrike'
elif target == 'high_plus50p':
    profitVar = 'profit50p'
elif target == 'high_plus30p':
    profitVar = 'profit30p'
elif target == 'high_plus10p':
    profitVar = 'profit'
filtered_df['weightedProfit'] = filtered_df[profitVar] * filtered_df['units']
filtered_mean = filtered_df.mean()
print('Profit margin: {} \nActing on {} stocks'.format(round(filtered_mean['weightedProfit'] / 10000,4),len(filtered_df)))

# confusion matrix 
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(predicted_df[target],np.where(predicted_df['prediction']>threshold,1,0))
print('Confusion Matrix : \n', cm1)
sensitivity1 = cm1[1,1]/(cm1[1,1]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

# Showing the ROC and AUC
from sklearn import metrics
auc = metrics.roc_auc_score(predicted_df[target], predicted_df['prediction'])
print("The Area under the ROC is {}".format(round(auc,3)))\
# to visualize
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(predicted_df[target], predicted_df['prediction'])
plt.plot(fpr, tpr)

# %%
data = yf.download('QD', start='2020-07-07', end='2020-07-21')
data
# %%
# IF happy save models
# LOGIT
logit_model.save('/Users/kasper.de-harder/gits/option_trading/modelLogitStock')

#%%
# RandomForest
# save the model to disk
filename = '/Users/kasper.de-harder/gits/option_trading/RandomForestStock.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# %%
