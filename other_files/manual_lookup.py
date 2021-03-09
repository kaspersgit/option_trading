# import packages
from dateutil.relativedelta import relativedelta, FR
import os
import sys
import pickle
import pandas as pd

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *

# model (disregard extension)
model = sys.argv[1]
model = model.split('.')[0]

ticker = sys.argv[2]
date = sys.argv[3]

# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/barchart/barchart_unusual_activity_{}.csv'.format(date)

# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# set working directory
current_path = os.getcwd()
file_path = current_path + '/trained_models/'+model+'.sav'
# load in model
with open(file_path, 'rb') as file:
	model = pickle.load(file)
model_name = file_path.split('/')[-1]
features = model.feature_names


# import data
#df = load_from_s3(profile="default", bucket=source_bucket, key_prefix=source_key)
try:
	df = pd.read_csv('data/barchart/barchart_unusual_activity_{}.csv'.format(date))
	print(df[df['baseSymbol'] == ticker])

	prob = model.predict_proba(df[features])[:, 1]

	print('Options contract scored')
	df['prediction'] = prob
	df['model'] = model_name

	print("Interesting columns")
	print(df[df['baseSymbol'] == ticker][['strikePrice','prediction']])
except:
	print('No scraped data found')

