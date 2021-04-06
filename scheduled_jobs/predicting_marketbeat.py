# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
import os, sys, platform
from datetime import datetime
import numpy as np
import pickle
import json
from option_trading_nonprod.aws import *

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

# model (disregard extension)
model = sys.argv[1]
model = model.split('.')[0]

# Set variagbles and load in data
day = day.strftime("%Y-%m-%d")
print('Mode: {}'.format(mode))
print('Model: {}'.format(model))
print('Using data from {}'.format(day))
# set working directory
os.chdir('/home/pi/Documents/python_scripts/option_trading')
current_path = os.getcwd()
df = pd.read_csv(current_path + '/data/marketbeat/marketbeat_call_activity_'+day+'.csv')

# Set source for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/marketbeat/marketbeat_call_activity_'+day+'.csv'
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# Get model which should be used
if platform.system() == 'Darwin':
	profile='mrOption'
else:
	profile='default'

df = load_from_s3(profile=profile, bucket=source_bucket, key_prefix=source_key)

# custom added variables
df['predDate'] = day
df['indicatorPresent'] = np.where(df['indicators'].isnull(),0,1)
df['upcomingEarning'] = np.where(df['indicators'].str.contains('Upcoming Earnings', na=False),1,0)
df['earningAnnounced'] = np.where(df['indicators'].str.contains('Earnings Announcement', na=False),1,0)
df['analystReport'] = np.where(df['indicators'].str.contains('Analyst Report', na=False),1,0)
df['heaveNewsReporting'] = np.where(df['indicators'].str.contains('Heavy News Reporting', na=False),1,0)
df['gapDown'] = np.where(df['indicators'].str.contains('Gap Down', na=False),1,0)
df['gapUp'] = np.where(df['indicators'].str.contains('Gap Up', na=False),1,0)

df['callStockVolume'] = df['avgStockVolume'] / df['avgOptionVolume']

# get last price on day of scraping
df['firstPrice'] = df['ticker'].map(getCurrentStockPrice)

with open('other_files/config_file.json') as json_file:
	config = json.load(json_file)

hprob_config = config['high_probability']

#%%
# Load model and predict
if model == 'LogisticRegression':
    # Logistic Regression
    file_path = current_path + '/trained_models/'+model
    model = LogitResults.load(file_path)
    model_name = file_path.split('/')[-1]
    # Select columns which are model needs as input but leave out the constant
    features = model.params.index
    prob = model.predict(df[features])
elif model != 'Logit':
	file_path = current_path + '/trained_models/'+model+'.sav'
	with open(file_path, 'rb') as file:
		model = pickle.load(file)
	model_name = file_path.split('/')[-1]
	features = model.feature_names
	prob = model.predict_proba(df[features])[:, 1]

print('Stocks scored')
df['prediction'] = prob
df['model'] = model_name

# %%
# Subsetting the predictions for highly probable stocks
threshold = 0.7
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 60
minStrikeIncrease = 1.05

virt_daysToExpiration = 21
df['expirationDate'] = (pd.to_datetime(df['dataDate']) + timedelta(days=virt_daysToExpiration)).dt.strftime('%Y-%m-%d')
df.rename(columns={'exportedAt': 'exportedAtTimestamp',
				   'dataDate': 'exportedAt',
				   'ticker': 'baseSymbol'},
		  inplace=True)

df['daysToExpiration'] = virt_daysToExpiration
df['baseLastPrice'] = df['firstPrice']
df['strikePrice'] = 1.1 * df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']


high_prob = df[(df['prediction'] > threshold)]
high_prob = high_prob[['baseSymbol', 'predDate', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction', 'model']]
high_prob = high_prob.sort_values('priceDiffPerc').reset_index(drop=True)

print('High probability table size: {}'.format(len(high_prob)))

# %%
# Sending an email with the predictions
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configurations and content
msg = MIMEMultipart()
msg['Subject'] = "Stock buy advise (MarketBeat)"
msg['From'] = 'k.sends.python@gmail.com'

html = """\
<html>
  <head></head>
  <body>
  	<h3> 10% increase in stock price </h3>
    {0}
    <h4> Configurations </h4>
    <p>
    Minimal threshold: {1} <br>
    Days to expiration: {2} <br>
    Stock price increase: {3} <br>
    </p>
  </body>
</html>
""".format(high_prob.to_html(),threshold,df['daysToExpiration'],df['priceDiffPerc']
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
