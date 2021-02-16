# import packages
from dateutil.relativedelta import relativedelta, FR
import os
import sys
import pandas as pd

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *

ticker = sys.argv[1]
date = sys.argv[2]

# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/barchart/barchart_unusual_activity_{}.csv'.format(date)

# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# import data
#df = load_from_s3(profile="default", bucket=source_bucket, key_prefix=source_key)
try:
	df = pd.read_csv('data/barchart/barchart_unusual_activity_{}.csv'.format(date))
	print(df[df['baseSymbol'] == ticker])
except:
	print('No scraped data found')

