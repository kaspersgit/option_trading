"""
This script should show the performance of all options which ended on the last Friday
import enriched options which ended friday from S3
score options with model
make summary (split per score band, graph,  TODO different durations, TODO estimate potential earnings)
command line command:
<script_name> <model> <mode>
example:
/home/pi/Documents/python_scripts/option_trading/scheduled_jobs/send_summary.py PROD_c_AB32_v1x2 DEVELOPMENT
"""

# import packages
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, FR
import os
import sys
import pickle
from sklearn.metrics import brier_score_loss

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *
from option_trading_nonprod.utilities.email import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.validation.classification import *
from option_trading_nonprod.process.stock_price_enriching import *


# Get supplied system arguments
# mode (development or production)
if len(sys.argv) >= 4:
	date = pd.to_datetime(sys.argv[3])
	last_friday = (date + relativedelta(weekday=FR(-1))).strftime('%Y-%m-%d')
else:
	last_friday = (datetime.today() + relativedelta(weekday=FR(-1))).strftime('%Y-%m-%d')

if len(sys.argv) >= 3:
	mode = sys.argv[2]
	if mode.upper().startswith('PROD'):
		mode = 'PRODUCTION'
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			emaillist = f.read().splitlines()
	elif mode.upper().startswith('DEV'):
		mode = 'DEVELOPMENT'
		with open('/home/pi/Documents/trusted/option_email_list_dev.txt') as f:
			emaillist = f.read().splitlines()

	# print status of variables
	print('Mode: {}'.format(mode))
	print('Emaillist: {}'.format(emaillist))

# Get model which should be used
model = sys.argv[1]
model = model.split('.')[0]

# Set wd and other variables
bucket = 'project-option-trading-output'
key = 'enriched_data/barchart/expired_on_{}.csv'.format(last_friday)

# print status of variables
print('Model : {}'.format(model))
print('Last Friday: {}'.format(last_friday))
print('Source bucket: {}'.format(bucket))
print('Source key: {}'.format(key))

# import data
df = load_from_s3(profile="default", bucket=bucket, key_prefix=key)
# df = pd.read_csv('/Users/kasper.de-harder/Downloads/exported_2021-01-19_expires_2021-01-29.csv')

print('Shape of imported data: {}'.format(df.shape))

# import model and score
file_path = os.getcwd() + '/trained_models/' + model + '.sav'
with open(file_path, 'rb') as file:
	model = pickle.load(file)
model_name = file_path.split('/')[-1]
features = model.feature_names
prob = model.predict_proba(df[features])[:, 1]

print('Loaded model and scored options')

# Make high level summary
# Add target variable
df['reachedStrikePrice'] = np.where(df['maxPrice'] >= df['strikePrice'], 1, 0)
# Add prediction variable
df['prob'] = prob

# Add columns
df['strikePricePerc'] = df['strikePrice'] / df['baseLastPrice']

# filter set on applicable rows
# only select Call option out of the money
optionType = 'Call'
minIncrease = 1.05

df = df[(df['symbolType'] == 'Call') & (df['strikePrice'] > df['baseLastPrice'] * minIncrease)]

# basic performance
# accuracy (split per days to expiration)
# accuracy (split per strike price increase)
# brier score
brier_score = brier_score_loss(df['reachedStrikePrice'], df['prob'])


print('Start creating plots')

# scatter plot
import matplotlib.pyplot as plt

ReachedStrike = df[df['reachedStrikePrice'] == 1]
notReachedStrike = df[df['reachedStrikePrice'] == 0]

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(notReachedStrike['strikePricePerc'], notReachedStrike['prob'], color='r', alpha=0.5,
		   label='Not reached strike')
ax.scatter(ReachedStrike['strikePricePerc'], ReachedStrike['prob'], color='g', alpha=0.5, label='Did reach strike')
ax.legend(loc="upper right")
ax.set_xlabel('Strike price increase')
ax.set_ylabel('Predicted probability')
ax.set_title('All Call options plotted')
plt.show()
fig.savefig("scheduled_jobs/summary_content/scatter.png")

print('Created and saved scatter plot')

# confusion matrix
# calibration curve
plotCalibrationCurve(df['reachedStrikePrice'], df['prob'], title='', bins=10, savefig=True,
					 saveFileName='scheduled_jobs/summary_content/CalibCurve.png')

print('Created and saved calibration plot')

# model performance
# AUC and similar
auc_roc = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='roc', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/roc.png')
auc_pr = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='pr', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/pr.png')

print('Created and saved AUC plot')
print('Composing email...')
# Send email
# recipient
# lay out and content
# attachment (the csv file)
html_content = """
<html>
  <head></head>
  <body>
	A summary of the call options expired last Friday and the models performs.
	<br>
	Showing only options of type: {}
	<br>
	With minimal price increase of: {}
	<br>
	Model used: {}
	<br><br>
	Total number of options (unique tickers): {} ({})
	<br>
	Options reaching strike (unique tickers): {} ({})
	<br>
	<br>
	Area Under Curve of ROC: {}
	<br>
	Brier loss score: {}




	<h3> Some graphs for visual interpretation</h3>
	<b> Plotting all options based on their profitability and probability </b>
	<br><img src="cid:image1"><br>


	<b> Plotting the calibration curve to see if the probabilities made sense </b>
	<br><img src="cid:image2"><br>


	<b> Plotting the ROC, which gives an idea on how well the model performs </b>
	<br><img src="cid:image3"><br>


	<b> Plotting the Precision Recall curve, which gives an idea on how well the model performs </b>
	<br><img src="cid:image4"><br>
	Looking good huh!
  </body>
""".format(optionType, minIncrease, model_name, len(df), df['baseSymbol'].nunique()
		   , len(ReachedStrike), ReachedStrike['baseSymbol'].nunique(), auc_roc, brier_score)
password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()
sendRichEmail(sender='k.sends.python@gmail.com'
			  , receiver=emaillist
			  , password=password
			  , subject='Performance report expiry date {}'.format(last_friday)
			  , content=html_content
			  , inline_images=['scheduled_jobs/summary_content/scatter.png', 'scheduled_jobs/summary_content/CalibCurve.png',
							   'scheduled_jobs/summary_content/roc.png', 'scheduled_jobs/summary_content/pr.png']
			  , attachment=None)


