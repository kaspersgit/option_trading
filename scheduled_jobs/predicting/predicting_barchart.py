# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults
import os, sys, platform
from datetime import datetime
import numpy as np
import pickle
import json

# to load custom packages
os.chdir("/home/pi/Documents/python_scripts/option_trading")
from option_trading_nonprod.aws import *
from option_trading_nonprod.process.stock_price_enriching import *

print('Script can be run from command line as <script> <model> <mode>')

# Get supplied system arguments
# mode (development or production)
if len(sys.argv) >= 3:
	mode = sys.argv[2]
	if mode.upper().startswith('PROD'):
		mode = 'PRODUCTION'
		# Load in todays scraped data
		day = datetime.today()
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			recipients = f.read().splitlines()
		emaillist = [elem.strip().split(',') for elem in recipients]
	elif mode.upper().startswith('DEV'):
		mode = 'DEVELOPMENT'
		# Load in scraped data of last business day
		day = datetime.today() - pd.tseries.offsets.BDay(1)
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			recipients = f.read().splitlines()
		emaillist = recipients[0]

# Set variables based on system we are running it on
if platform.system() == 'Darwin':
	profile='mrOption'
	os.chdir('/Users/kasper.de-harder/gits/option_trading')
else:
	profile='default'
	os.chdir('/home/pi/Documents/python_scripts/option_trading')

# set working directory
current_path = os.getcwd()

# model (disregard extension)
model_name = sys.argv[1]
model_name = model_name.split('.')[0]

# print current timestamp for logging
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Set variagbles and load in data
day = day.strftime("%Y-%m-%d")
print('Mode: {}'.format(mode))
print('Model: {}'.format(model_name))
print('Using data from {}'.format(day))

# Set source for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/barchart/barchart_unusual_activity_'+day+'.csv'
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# Get data
df = load_from_s3(profile=profile, bucket=source_bucket, key_prefix=source_key)

print(f"Imported dataframe shape: {df.shape}")

# print current timestamp for logging
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Adding most recent stock price
unique_tickers = df['baseSymbol'].unique()
live_price = getCurrentStockPrice(unique_tickers, attribute='Close')
live_price.rename(columns={'ticker': 'baseSymbol',
						   'livePrice': 'baseLivePrice'}, inplace=True)

df = pd.merge(df, live_price, on='baseSymbol', how='left')
df.rename(columns={'baseLastPrice': 'baseLastPriceScrape',
				   'baseLivePrice': 'baseLastPrice'}, inplace=True)

df = df[~df['baseLastPrice'].isnull()]

print("Added latest stock prices")
print("Shape of dataframe: {}".format(df.shape))

if mode == 'DEVELOPMENT':
	from option_trading_nonprod.process.stock_price_enriching import *
	df = batch_enrich_df(df)

with open('other_files/config_file.json') as json_file:
	config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']

# Adding some additional columns
df['predDate'] = day
df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
df['inTheMoney'] = np.where(df['baseLastPrice'] >= df['strikePrice'],1,0)
df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','inTheMoney'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'
        ]).agg({'baseSymbol':'count', 'strikePrice':'mean'
        }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
        }).reset_index()
df = pd.merge(df,df_symbol, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'])

df['const'] = 1.0

#%%
# Load model and predict
if model_name == 'LogisticRegression':
    # Logistic Regression
    file_path = current_path + '/trained_models/'+model_name
    model = LogitResults.load(file_path)
    # Select columns which are model needs as input but leave out the constant
    features = model.params.index
    prob = model.predict(df[features])
elif model_name != 'Logit':
	file_path = current_path + '/trained_models/'+model_name+'.sav'
	with open(file_path, 'rb') as file:
		model = pickle.load(file)
	features = model.feature_names
	prob = model.predict_proba(df[features])[:, 1]

print('Options contract scored')

# print current timestamp for logging
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

df['prediction'] = prob
df['model'] = model_name


# %%
# Subsetting the predictions for highly probable stocks
threshold = 0.7
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 25
minStrikeIncrease = 1.05

high_prob = df[(df['prediction'] > hprob_config['minThreshold']) &
    (df['symbolType']=='Call') &
    (df['daysToExpiration'] < hprob_config['maxDaysToExp']) &
    (df['priceDiffPerc'] > hprob_config['minStrikeIncrease']) &
    (df['daysToExpiration'] > hprob_config['minDaysToExp']) &
    (df['baseLastPrice'] < hprob_config['maxBasePrice'])].copy()
high_prob = high_prob[['baseSymbol', 'predDate', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction', 'model']]
high_prob = high_prob.sort_values('priceDiffPerc').reset_index(drop=True)

print('High probability table size: {}'.format(len(high_prob)))

# Subsetting the predictions for highly profitable stocks
hprof_threshold = 0.25
hprof_maxBasePrice = 1000
hprof_minDaysToExp = 3
hprof_maxDaysToExp = 25
hprof_minStrikeIncrease = 1.20

high_prof = df[(df['prediction'] > hprof_config['minThreshold']) &
    (df['symbolType']=='Call') &
    (df['daysToExpiration'] < hprof_config['maxDaysToExp']) &
    (df['priceDiffPerc'] > hprof_config['minStrikeIncrease']) &
    (df['daysToExpiration'] > hprof_config['minDaysToExp']) &
    (df['baseLastPrice'] < hprof_config['maxBasePrice'])].copy()
high_prof = high_prof[['baseSymbol', 'predDate', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction', 'model']]
high_prof = high_prof.sort_values('priceDiffPerc').reset_index(drop=True)

print('High profitability table size: {}'.format(len(high_prof)))

# %%
# Sending an email with the predictions
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configurations and content
msg = MIMEMultipart()
msg['Subject'] = "Stock buy advise"
msg['From'] = 'k.sends.python@gmail.com'

html = """\
<html>
  <head></head>
  <body>
  	<h3> High probability </h3>
    {0}
    <h4> Configurations </h4>
    <p>
    Minimal threshold: {1} <br>
    Maximum stock price: {2} <br>
    Days to expiration between {3} and {4} <br>
    Strike price at least {5} higher than stock price <br>
    </p>
    <hr>
    <h3> High profitability </h3>
    {6}
    <h4> Configurations </h4>
    <p>
    Minimal threshold: {7} <br>
    Maximum stock price: {8} <br>
    Days to expiration between {9} and {10} <br>
    Strike price at least {11} higher than stock price <br>
    </p>
  </body>
</html>
""".format(high_prob.to_html(),hprob_config['minThreshold'],hprob_config['maxBasePrice'],
		   hprob_config['minDaysToExp'],hprob_config['maxDaysToExp'],hprob_config['minStrikeIncrease'],
		   high_prof.to_html(),hprof_config['minThreshold'],hprof_config['maxBasePrice'],
		   hprof_config['minDaysToExp'],hprof_config['maxDaysToExp'],hprof_config['minStrikeIncrease'],
		   )

part1 = MIMEText(html, 'html')
msg.attach(part1)

# Sending the email
import smtplib, ssl

port = 465  # For SSL
password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(msg['From'], password)
    server.sendmail(msg['From'], emaillist , msg.as_string())

print('Email with predictions send')
