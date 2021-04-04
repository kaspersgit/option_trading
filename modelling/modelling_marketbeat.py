#%%
# Load in all csv files from source folder
import os
import pandas as pd
from datetime import datetime, timedelta
from option_trading_nonprod.process import *
from option_trading_nonprod.aws import *
from option_trading_nonprod.process.stock_price_enriching import *

# Temp setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###### import data from S3
# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/marketbeat/'
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))


df = load_from_s3(profile="mrOption", bucket=source_bucket, key_prefix=source_key)
print("Raw imported data shape: {}".format(df.shape))
######

# Delete duplicates
df = df.drop_duplicates(subset=['ticker','exportedAt'], keep='first')
print("After dropping duplicates: {}".format(df.shape))

# make fake expiration date to let function work
virt_daysToExpiration = 21
df['expirationDate'] = (pd.to_datetime(df['dataDate']) + timedelta(days=virt_daysToExpiration)).dt.strftime('%Y-%m-%d')
df.rename(columns={'dataDate': 'exportedAt',
                   'ticker': 'baseSymbol'},
          inplace=True)
df['daysToExpiration'] = virt_daysToExpiration

# filter on only mature options
cutoff_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
print('Cutoff date used: {}'.format(cutoff_date))

df = df[df['expirationDate'] < cutoff_date]
print("After filtering on the cutoff date: {}".format(df.shape))

# Get min max first and last price
# the thrown chunkedencodingerror might be temp solved by https://github.com/SamPom100/UnusualVolumeDetector/issues/22
df_ = df[~df.baseSymbol.isin(['NVCR','UA','ADMA'])]
contracts_prices = getContractPrices(df_, startDateCol='exportedAt', endDateCol='expirationDate', type='minmax')

# incase it goes wrong somewhere, start from close to that row
# df_last_few = df.drop_duplicates(subset=['baseSymbol'], keep='first')
# df_last_few = df_last_few.iloc[2099::]
# df_last_few = df_last_few.head(1)

# Get technical indicators
# Get stock prices from 35 days before export date to calculate them
df['exportedAt'] = pd.to_datetime(df['exportedAt'])
df['start_date'] = df['exportedAt'] - timedelta(days=45)
indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')


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
