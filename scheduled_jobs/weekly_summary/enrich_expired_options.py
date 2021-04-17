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
	date = pd.to_datetime(sys.argv[1])
	last_friday = (date + relativedelta(weekday=FR(-1)))
else:
	print('Script can be run from command line as <script> <date (YYYY-MM-DD)>')
	last_friday = (datetime.today() + relativedelta(weekday=FR(-1)))

# import data
if platform.system() == 'Darwin':
	s3_profile = 'mrOption'
else:
	s3_profile = 'default'

# Set bucket
bucket = 'project-option-trading'

# Get all dates from last friday until saturday previous
numdays = 7
date_list = [(last_friday - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(numdays)]

# check which what expiration date(s) are present
s3_client = connect_to_s3(s3_profile, type="client")

for d in date_list:
	possible_key = 'on_expiry_date/expires_{}.csv'.format(d)
	exist, key = get_s3_key(s3_client, bucket, possible_key)
	if exist:
		break

# Set source and target for bucket and keys
output_bucket = 'project-option-trading-output'
output_key = 'enriched_data/barchart/expired_on_{}.csv'.format(d)

# print status of variables
print('Last Friday: {}'.format(d))
print('Source bucket: {}'.format(bucket))
print('Source key: {}'.format(key))
print('Output bucket: {}'.format(output_bucket))
print('Output key: {}'.format(output_key))

df = load_from_s3(profile=s3_profile, bucket=source_bucket, key_prefix=key)

# Delete duplicates
df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'], keep='first')

print('Shape of imported data: {}'.format(df.shape))

# enrich df
print('Enriching stocks...')
contracts_prices = getContractPrices(df)

# Changed to fit format of contract prices to be able to merge
df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')

# Put dfs together to have all enriched data
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
print('Enriching stocks...Done')

# Upload enriched table to S3
write_dataframe_to_csv_on_s3(profile=s3_profile, dataframe=df_enr, filename=output_key, bucket=output_bucket)