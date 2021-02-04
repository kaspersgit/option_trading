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
import sys
import pickle

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.validation.classification import *
from option_trading_nonprod.process.stock_price_enriching import *

# Get supplied system arguments
# mode (development or production)
if len(sys.argv) >= 3:
	mode = sys.argv[2]
	if mode.upper().startswith('PROD'):
		mode = 'PRODUCTION'
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			recipients = f.read().splitlines()
		emaillist = [elem.strip().split(',') for elem in recipients]
	elif mode.upper().startswith('DEV'):
		mode = 'DEVELOPMENT'
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			recipients = f.read().splitlines()
		emaillist = recipients[0]

	# print status of variables
	print('Mode: {}'.format(mode))
	print('Emaillist: {}'.format(emaillist))

# Get model which should be used
model = sys.argv[1]
model = model.split('.')[0]

# Set wd and other variables
last_friday = (datetime.today()
    - timedelta(days=datetime.today().weekday())
    + timedelta(days=4, weeks=-1)).strftime('%Y-%m-%d')
bucket = 'project-option-trading'
key = f'on_expiry_date/expires_{last_friday}/'

# print status of variables
print('Model : {}'.format(model))
print('Last Friday: {}'.format(last_friday))
print('Source bucket: {}'.format(bucket))
print('Source key: {}'.format(key))

# import data
df = load_from_s3(profile="default", bucket=bucket, key_prefix=key)
# df = pd.read_csv('/Users/kasper.de-harder/Downloads/exported_2021-01-19_expires_2021-01-29.csv')

print('Shape of imported data: {}'.format(df.shape))

# clean and format data
# df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')

# enrich df
print('Enriching stocks...')
contracts_prices = getContractPrices(df)

# Put dfs together to have all enriched data
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
print('Enriching stocks...Done')

# import model and score
file_path = os.getcwd() + '/trained_models/' + model + '.sav'
with open(file_path, 'rb') as file:
	model = pickle.load(file)
model_name = file_path.split('/')[-1]
features = model.feature_names
prob = model.predict_proba(df_enr[features])[:, 1]

print('Loaded model and scored options')

# Make high level summary
# Add target variable
df_enr['reachedStrikePrice'] = np.where(df_enr['maxPrice'] >= df_enr['strikePrice'],1,0)
# Add prediction variable
df_enr['prob'] = prob
# filter set on applicable rows
# only select Call option out of the money
df_enr = df_enr[(df_enr['symbolType']=='Call') & (df_enr['strikePrice'] > df_enr['baseLastPrice'])]

# Add columns
df_enr['strikePricePerc'] = df_enr['strikePrice'] / df_enr['baseLastPrice']

# basic performance
# accuracy (split per days to expiration)
# accuracy (split per strike price increase)

print('Start creating plots')

# scatter plot
import matplotlib.pyplot as plt
ReachedStrike = df_enr[df_enr['reachedStrikePrice']==1]
notReachedStrike = df_enr[df_enr['reachedStrikePrice']==0]

fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(notReachedStrike['strikePricePerc'], notReachedStrike['prob'], color='r', alpha=0.5, label='Not reached strike')
ax.scatter(ReachedStrike['strikePricePerc'], ReachedStrike['prob'], color='g', alpha=0.5, label='Did reach strike')
ax.legend(loc="upper right")
ax.set_xlabel('Strike price increase')
ax.set_ylabel('Predicted probability')
ax.set_title('All Call options plotted')
plt.show()
fig.savefig("validation/scatter.png")

print('Created and saved scatter plot')

# confusion matrix
# calibration curve
plotCalibrationCurve(df_enr['reachedStrikePrice'], df_enr['prob'], title='', bins=10, savefig=True, saveFileName='validation/CalibCurve.png')

print('Created and saved calibration plot')

# model performance
# AUC and similar
auc_roc = plotCurveAUC(df_enr['prob'],df_enr['reachedStrikePrice'], title='', type='roc', savefig=True, saveFileName='validation/roc.png')

print('Created and saved AUC plot')
print('Composing email...')
# Send email
# recipient
# lay out and content
# attachment (the csv file)
html_content ="""
<html>
  <head></head>
  <body>
	A summary of the call options expired last Friday and the models performs.
	Only call options being out of the money at moment of scraping are included.
	<br><br>Total number of options (unique tickers): {} ({})
	<br>Options reaching strike (unique tickers): {} ({})
	
	<br>Model used: {}
	

	<h3> Some graphs for visual interpretation</h3>
	<b> Plotting all options based on their profitability and probability </b>
	<br><img src="cid:image1"><br>
	
	
	<b> Plotting the calibration curve to see if the probabilities made sense </b>
	<br><img src="cid:image2"><br>
	
	
	<b> Plotting the ROC, which gives an idea on how well the model performs </b>
	<br><img src="cid:image3"><br>
	Looking good huh!
  </body>
""".format(len(df_enr), df_enr['baseSymbol'].nunique()
		   , len(ReachedStrike), ReachedStrike['baseSymbol'].nunique(), model_name)
password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()
sendRichEmail(sender='k.sends.python@gmail.com'
			  , receiver = emaillist
			  , password = password
			  , subject = 'Performance report expiry date {}'.format(last_friday)
			  , content = html_content
			  , inline_images = ['validation/scatter.png','validation/CalibCurve.png','validation/roc.png']
			  , attachment = None)


