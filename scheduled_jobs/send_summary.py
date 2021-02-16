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
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

os.chdir("/home/pi/Documents/python_scripts/option_trading")

from option_trading_nonprod.aws import *
from option_trading_nonprod.other.trading_strategies import *
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

else:
	print('Script can be run from command line as <script> <model> <env prod or dev> <date (optional)>')

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
# df = pd.read_csv('/Users/kasper.de-harder/Downloads/expired_on_2021-02-05.csv')

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

# progress towards strikeprice
df['strikePriceProgress'] = (df['maxPrice'] - df['baseLastPrice']) / (df['strikePrice'] - df['baseLastPrice'])
df['strikePriceProgressCapped'] = np.where(df['strikePriceProgress'] >= 1, 1.0, df['strikePriceProgress'])
# Add columns
df['strikePricePerc'] = df['strikePrice'] / df['baseLastPrice']
# expected profit
df['expPercIncrease'] = df['strikePricePerc'] * df['prob']
# profitability (percentage increase from stock price to max price)
df['profitability'] = df['maxPrice']/df['baseLastPrice']

# filter set on applicable rows
# only select Call option out of the money
optionType = 'Call'
minIncrease = 1.05
maxIncrease = 2
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 25

df = df[(df['symbolType'] == optionType) & (df['strikePrice'] > df['baseLastPrice'] * minIncrease) & (df['strikePricePerc'] < maxIncrease)]

# Basic summary
# Get top performing stocks (included/not included in email)
biggest_increase_df = df.sort_values('profitability', ascending=False)[['baseSymbol','exportedAt','baseLastPrice','strikePrice','maxPrice','maxPriceDate','profitability','prob']].drop_duplicates(subset=['baseSymbol']).head(5)
biggest_increase_df.reset_index(drop=True, inplace=True)
# biggest_increase_df['in_email'] = np.where()


# basic performance
# accuracy (split per days to expiration)
# accuracy (split per strike price increase)
# brier score
brier_score = brier_score_loss(df['reachedStrikePrice'], df['prob'])

# Simulating trading strategies
print('Simulating simple trading strategies')

# buy one of each stock of a certain amount worth of stocks (df['baseLastPrice'] vs 100)
# below we implement 100 dollars worth of each stock
df['stocksBought'] = 100 / df['baseLastPrice']
df['cost'] = df['stocksBought'] * df['baseLastPrice']
df['revenue'] = df['stocksBought'] * np.where(df['reachedStrikePrice'] == 1, df['strikePrice'], df['finalPrice'])
df['profit'] = df['revenue'] - df['cost']

filterset = {'threshold': 0.7,
			 'maxBasePrice': 200,
			 'minStrikeIncrease': 1.05,
			 'minDaysToExp': 3,
			 'maxDaysToExp': 25}
roi_highprob, cost_highprob, revenue_highprob, profit_highprob = simpleTradingStrategy(df,filterset, plot=False)


filterset = {'threshold': 0.25,
			 'maxBasePrice': 100,
			 'minStrikeIncrease': 1.2,
			 'minDaysToExp': 3,
			 'maxDaysToExp': 25}
roi_highprof, cost_highprof, revenue_highprof, profit_highprof = simpleTradingStrategy(df,filterset, plot=False)

print('Start creating plots')

# scatter plot (strike percentage increase against predicted probability)
ReachedStrike = df[df['reachedStrikePrice'] == 1]
notReachedStrike = df[df['reachedStrikePrice'] == 0]

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(ReachedStrike['strikePricePerc'], ReachedStrike['prob'], s = 7, color='g', alpha=0.7, label='Did reach strike')
ax.scatter(notReachedStrike['strikePricePerc'], notReachedStrike['prob'], s = 7, color='r', alpha=0.7,
		   label='Not reached strike')
ax.legend(loc="upper right")
ax.set_xlabel('Strike price increase')
ax.set_ylabel('Predicted probability')
ax.set_title('All Call options plotted')
plt.show()
fig.savefig("scheduled_jobs/summary_content/scatter.png")

print('Created and saved scatter plot (percentage increase vs predicted probability')

#################################### Unsure
# Create scatter plot (strike price progress vs predicted probability)
# filter to make plot readable
df_plot = df[(df['strikePriceProgressCapped'] < 1.1) & (df['strikePricePerc'] > 1.0) & (df['strikePricePerc'] < 10)]
df_plot = df
fig = plt.figure()
# cm = plt.cm.get_cmap('Blues')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
im = ax.scatter(df_plot['strikePriceProgressCapped'], df_plot['expPercIncrease'], s=20, alpha=0.7)
# fig.colorbar(im, ax=ax)
ax.set_xlabel('% of strike price reached')
ax.set_ylabel('Expected profit')
ax.set_title('All Call options plotted')
plt.show()
fig.savefig("scheduled_jobs/summary_content/scatter_strikeProgress.png")

print('Created and saved scatter plot (percentage increase vs predicted probability')
##############################

# confusion matrix
# calibration curve
plotCalibrationCurve(df['reachedStrikePrice'], df['prob'], title='', bins=10, savefig=True,
					 saveFileName='scheduled_jobs/summary_content/CalibCurve.png')

print('Created and saved calibration plot')

# model performance
# AUC and similar
auc_roc = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='roc', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/roc.png')

print('Created and saved ROC plot')

auc_pr = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='pr', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/pr.png')

print('Created and saved Precision Recall plot')

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
	With minimal price increase between: {} and {}
	<br>
	Model used: {}
	<br><br>
	Total number of options (unique tickers): {} ({})
	<br>
	Options reaching strike (unique tickers): {} ({})
	<br><br>
	<hr>
	<h3>Model performance metrics</h3>
	
	Area Under Curve of ROC: 	{}
	<br>
	AUC of Precision Recall: 	{}
	<br>
	Brier loss score: 			{}
	<br><br>
	<hr>



	<h3> Some graphs for visual interpretation</h3>
	Plotting all options based on their profitability and probability
	<br><img src="cid:image1"><br>


	Plotting the calibration curve to see if the probabilities made sense
	<br><img src="cid:image2"><br>


	Plotting the ROC, which gives an idea on how well the model performs
	<br><img src="cid:image3"><br>

	<br><br>
	<hr>
	<h3>Most profitable stocks</h3>
	<br>
	{}
	<br><br>
	<h3>Implementing a simple trading strategy</h3>
	<br>
	Purely buying selling stocks which are mentioned in the email
	<br>
	Return on investment (ROI):
	<br>
	High probability: {}
	<br>
	High profitability: {}
	
  </body>
""".format(optionType, minIncrease, maxIncrease, model_name, len(df), df['baseSymbol'].nunique()
		   , len(ReachedStrike), ReachedStrike['baseSymbol'].nunique()
		   , round(auc_roc,3), round(auc_pr,3) , round(brier_score,3)
		   , biggest_increase_df, ound(roi_highprob,3), round(roi_highprof,3))
password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()
sendRichEmail(sender='k.sends.python@gmail.com'
			  , receiver=emaillist
			  , password=password
			  , subject='Performance report expiry date {}'.format(last_friday)
			  , content=html_content
			  , inline_images=['scheduled_jobs/summary_content/scatter.png', 'scheduled_jobs/summary_content/CalibCurve.png',
							   'scheduled_jobs/summary_content/roc.png']
			  , attachment=df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice',
							   'expirationDate', 'lastPrice', 'exportedAt',
							   'finalPrice', 'finalPriceDate', 'firstPrice', 'firstPriceDate',
							   'maxPrice', 'maxPriceDate', 'minPrice', 'minPriceDate',
							   'prob']].reset_index(drop=True)
)


