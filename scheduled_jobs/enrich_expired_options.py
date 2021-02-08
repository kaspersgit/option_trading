"""
This script should show the performance of all options which ended on the last Friday
import options which ended friday from S3
Enrich options with stock price data
"""

# import packages
from dateutil.relativedelta import relativedelta, FR
import os
import sys
import pandas as pd

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *
from option_trading_nonprod.process.stock_price_enriching import *

# Get supplied system arguments
if len(sys.argv) >= 2:
	date = pd.to_datetime(sys.argv[2])
	last_friday = (date + relativedelta(weekday=FR(-1))).strftime('%Y-%m-%d')
else:
	last_friday = (datetime.today() + relativedelta(weekday=FR(-1))).strftime('%Y-%m-%d')



# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'on_expiry_date/expires_{}/'.format(last_friday)
output_bucket = 'project-option-trading-output'
output_key = 'enriched_data/barchart/expired_on_{}.csv'.format(last_friday)

# print status of variables
print('Last Friday: {}'.format(last_friday))
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))
print('Output bucket: {}'.format(output_bucket))
print('Output key: {}'.format(output_key))

# import data
df = load_from_s3(profile="default", bucket=source_bucket, key_prefix=source_key)
# df = pd.read_csv('/Users/kasper.de-harder/Downloads/exported_2021-01-19_expires_2021-01-29.csv')

print('Shape of imported data: {}'.format(df.shape))

# enrich df
print('Enriching stocks...')
contracts_prices = getContractPrices(df)

# Put dfs together to have all enriched data
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
print('Enriching stocks...Done')

# Upload enriched table to S3
write_dataframe_to_csv_on_s3(profile="default", dataframe=df_enr, filename=output_key, bucket=output_bucket)