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
source_key = 'raw_data/barchart/'
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# import data
if platform.system() == 'Darwin':
    s3_profile = 'mrOption'
else:
    s3_profile = 'default'

df = load_from_s3(profile=s3_profile, bucket=source_bucket, key_prefix=source_key)
print("Raw imported data shape: {}".format(df.shape))
######

# Data mature until 10 days ago
cutoff_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
print('Cutoff date used: {}'.format(cutoff_date))

# Delete duplicates
df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'], keep='first')
print("After dropping duplicates: {}".format(df.shape))

# Select columns
df = df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice', 'expirationDate', 'daysToExpiration', 'bidPrice', 'midpoint', 'askPrice', 'lastPrice', 'volume', 'openInterest', 'volumeOpenInterestRatio', 'volatility', 'tradeTime', 'exportedAt']]

# filter on only mature options
df = df[df['expirationDate'] < cutoff_date]
print("After filtering on the cutoff date: {}".format(df.shape))

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
minDaysToExp = 3
maxDaysToExp = 60
df = limitDaysToExpiration(df, min=minDaysToExp, max=maxDaysToExp)
print("After filtering on having days to expiration between {} and {} \nThe left over data shape: {}".format(minDaysToExp, maxDaysToExp, df.shape))

# Get min max first and last price
contracts_prices = getContractPrices(df, startDateCol='exportedAt', endDateCol='expirationDate', type='minmax')

# incase it goes wrong somewhere, start from close to that row
# df_last_few = df.drop_duplicates(subset=['baseSymbol'], keep='first')
# df_last_few = df_last_few.iloc[2099::]
# df_last_few = df_last_few.head(1)

# Get technical indicators (can't be used on raspberry in current form)
# Get stock prices from 35 days before export date to calculate them
# df['exportedAt'] = pd.to_datetime(df['exportedAt'])
# df['start_date'] = df['exportedAt'] - timedelta(days=45)
# indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')

# to make sure the join columns are of same type
df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')
df['expirationDate'] = pd.to_datetime(df['expirationDate']).dt.strftime('%Y-%m-%d')
contracts_prices['exportedAt'] = pd.to_datetime(contracts_prices['exportedAt']).dt.strftime('%Y-%m-%d')
contracts_prices['expirationDate'] = pd.to_datetime(contracts_prices['expirationDate']).dt.strftime('%Y-%m-%d')

# Put dfs together
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
# df_enr = df_enr.merge(indicators_df, on=['baseSymbol','exportedAt'])

today = datetime.today().strftime('%Y-%m-%d')
output_bucket = 'project-option-trading-output'
output_key = 'train_data/barchart/enriched_on_{}.csv'.format(today)

print('Source bucket: {}'.format(output_bucket))
print('Source key: {}'.format(output_key))

# Upload enriched table to S3
write_dataframe_to_csv_on_s3(profile=s3_profile, dataframe=df_enr, filename=output_key, bucket=output_bucket)