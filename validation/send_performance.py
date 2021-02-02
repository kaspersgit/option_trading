"""
This script should show the performance of all options which ended on the last Friday
import options which ended friday from S3
Add stock price data
score options with model
make summary (split per score band, graph, different durations, estimate potential earnings)
"""

# import packages
import boto3
import pandas as pd
from datetime import datetime, timedelta
import os

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *
from option_trading_nonprod.process.stock_price_enriching import *

# Set wd and other variables
last_friday = (datetime.today()
    - timedelta(days=datetime.today().weekday())
    + timedelta(days=4, weeks=-1)).strftime('%Y-%m-%d')
bucket = 'project-option-trading'
key = f'on_expiry_date/expires_{last_friday}/'

# import data
df = load_from_s3(profile="default", bucket=bucket, key_prefix=key)

# clean and form data
df['exportedAt']

# enrich df
# Using above functions
contracts_prices = getContractPrices(df)

# Put dfs together
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])