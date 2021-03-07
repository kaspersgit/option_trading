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

# Set path
cwd = os.getcwd()
directory = os.path.join(cwd,"data/barchart")

# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/barchart/'

#### from local
# create empty df
df = pd.DataFrame()
# loop through all csv files and concatenate
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           f = pd.read_csv(os.path.join(root,file))
           #  Concatenate files into one pandas df
           df = pd.concat([df,f])
df.reset_index(drop=True, inplace=True)
#####



###### import data from S3
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))


df = load_from_s3(profile="default", bucket=source_bucket, key_prefix=source_key)
######

# Data mature until 10 days ago
cutoff_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
print('Cutoff date used: {}'.format(cutoff_date))

# Delete duplicates
df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'], keep='first')

# Select columns
df = df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice', 'expirationDate', 'daysToExpiration', 'bidPrice', 'midpoint', 'askPrice', 'lastPrice', 'volume', 'openInterest', 'volumeOpenInterestRatio', 'volatility', 'tradeTime', 'exportedAt']]

# filter on only mature options
df = df[df['expirationDate'] < cutoff_date]

# Helper functions
def colType2Float(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    return clean_series

def colType2Int(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    clean_series = clean_series.astype('int')
    return clean_series

# Cast certain columns to float after cleaning
float_cols = ['baseLastPrice','strikePrice','bidPrice','midpoint','askPrice','lastPrice','volumeOpenInterestRatio','volatility']
for col in float_cols:
    df[col] = colType2Float(df[col])

# Changing date columns to datetime
date_cols = ['expirationDate','tradeTime','exportedAt']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Cast columns to int
int_cols = ['volume','openInterest']
for col in int_cols:
    df[col] = colType2Int(df[col])

# add target label
# stock price reached strike price at expiration date

# Filter df
# on only short time to expiration
df = limitDaysToExpiration(df, min=3, max=60)
# Delete duplicates
df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])


# Using above functions
contracts_prices = getContractPrices(df, startDateCol='exportedAt', endDateCol='expirationDate', type='minmax')

# Get technical indicators
# Get stock prices from 35 days before export date to calculate them
df['exportedAt'] = pd.to_datetime(df['exportedAt'])
df['start_date'] = df['exportedAt'] - timedelta(days=35)
indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')

# Put dfs together
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
df_enr = df_enr.merge(indicators_df, on=['baseSymbol','exportedAt'])

# Save enriched df as csv
# x2 as we add technical indicators here
df_enr.to_csv('data/barchart_yf_enr_1x2.csv')
