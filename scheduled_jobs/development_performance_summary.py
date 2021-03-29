# A summary sendout which focusses more on the model performance
# more into detail to understand where performance is lacking
# and profitability is low

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
from dateutil.relativedelta import relativedelta, FR
import os, sys, platform
import pickle
import json

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

# Set mode prod or dev and base email recipients on that
# decide if attachment should be send or not
if len(sys.argv) >= 3:
	mode = sys.argv[2]
	if mode.upper().startswith('PROD'):
		mode = 'PRODUCTION'
		add_attachment = True
		with open('/home/pi/Documents/trusted/option_predict_email_receivers.txt') as f:
			emaillist = f.read().splitlines()
	elif mode.upper().startswith('DEV'):
		mode = 'DEVELOPMENT'
		add_attachment = False
		with open('/home/pi/Documents/trusted/option_email_list_dev.txt') as f:
			emaillist = f.read().splitlines()

	# print status of variables
	print('Mode: {}'.format(mode))
	print('Emaillist: {}'.format(emaillist))

else:
	print('Script can be run from command line as <script> <model> <env prod or dev> <add attachment true/false> <date (optional)>')

# Get model which should be used
if platform.system() == 'Darwin':
	model = 'DEV_c_GB64_v1x3'
else:
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
if platform.system() == 'Darwin':
	s3_profile = 'mrOption'
else:
	s3_profile = 'default'

df = load_from_s3(profile=s3_profile, bucket=bucket, key_prefix=key)
# df = pd.read_csv('/Users/kasper.de-harder/Downloads/expired_on_2021-02-05.csv')
# enrich data within batches
df = batch_enrich_df(df)

print('Shape of imported data: {}'.format(df.shape))

# import model and score
file_path = os.getcwd() + '/trained_models/' + model + '.sav'
with open(file_path, 'rb') as file:
	model = pickle.load(file)
model_name = file_path.split('/')[-1]
features = model.feature_names
prob = model.predict_proba(df[features])[:, 1]

print('Loaded model and scored options')

####### Adding columns ########################
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
df['maxProfitability'] = df['maxPrice']/df['baseLastPrice']

# buy one of each stock of a certain amount worth of stocks (df['baseLastPrice'] vs 100)
# below we implement 100 dollars worth of each stock
df['stocksBought'] = 100 / df['baseLastPrice']
df['cost'] = df['stocksBought'] * df['baseLastPrice']
df['revenue'] = df['stocksBought'] * np.where(df['reachedStrikePrice'] == 1, df['strikePrice'], df['finalPrice'])
df['profit'] = df['revenue'] - df['cost']
df['profitPerc'] = df['profit'] / df['cost']
################################################

# filter set on applicable rows
# only select Call option out of the money
with open('other_files/config_file.json') as json_file:
	config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']

# Filter on basics (like days to expiration and contract type)
df = df[(df['symbolType'] == hprob_config['optionType']) &
		(df['daysToExpiration'] >= hprob_config['minDaysToExp']) &
		(df['daysToExpiration'] < hprob_config['maxDaysToExp']) &
		(df['priceDiffPerc'] > hprob_config['minStrikeIncrease']) &
		(df['baseLastPrice'] < hprob_config['maxBasePrice'])]




# Calibration plot
title = 'all test observations - {}'.format(model_name)
plotCalibrationCurve(df['reachedStrikePrice'], df['prob'], title=title, bins=10)

# Show performance for different segments
brackets = [{'lower':1.05, 'higher':1.07}
	, {'lower':1.07, 'higher':1.15}
	, {'lower':1.15, 'higher':1.4}
	, {'lower':1.4, 'higher':9999}
			]

for bracket in brackets:
	print('Strike price / stock price ratio between {} and {}'.format(bracket['lower'], bracket['higher']))

	title = '{} with ratio between {} and {}'.format(model_name, bracket['lower'], bracket['higher'])
	select_df = df[(df['priceDiffPerc'] >= bracket['lower']) & (df['priceDiffPerc'] < bracket['higher'])].copy()
	print('Nr observations {} out of {} ({})'.format(len(select_df),len(df),round(len(select_df)/len(df),2)))

	# auc_roc = plotCurveAUC(select_df['prob'],select_df['reachedStrikePrice'], title = title,type='roc')
	auc_pr = plotCurveAUC(select_df['prob'],select_df['reachedStrikePrice'], title = title,type='pr')
	# print('AUC ROC: {}'.format(round(auc_roc,3)))
	print('AUC PR: {}'.format(round(auc_pr,3)))
	# plotCalibrationCurve(select_df['reachedStrikePrice'], select_df['prob'], title = title, bins=10)

	# simpleTradingStrategy(select_df, actualCol='reachedStrikePrice', filterset={'minThreshold': 0.1, 'maxThreshold': 0.99, 'minDaysToExp': 3, 'maxDaysToExp': 60, 'minStrikeIncrease': 1.05}, title = title)

### understanding low profit high auc
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 150)

select_df[['baseSymbol','baseLastPrice','strikePrice','maxPrice','finalPrice','prob', 'reachedStrikePrice', 'profit']].sort_values('prob')

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
	<hr>
	<h3>Implementing a simple trading strategy</h3>
	<br>
	Purely buying selling stocks which are mentioned in the email
	<br>
	Return on investment (ROI):
	<br>
	High probability: {}
	<br>
	High profitability: {}
	<br><br>
	<hr>
	
	Plotting expected profitability vs actual profitability
	<small> 
	Expected: (difference in stock and strike price * predicted probability) / stock price
	<br>
	Actual:	(difference in either strike price (if reached) or stock price on close before expiration and stock price) / stock price
	</small>
	<br><img src="cid:image4"><br>
	
	Plotting expected profitability vs max profitability
	<small> 
	Expected: (difference in stock and strike price * predicted probability) / stock price
	<br>
	Max:	(difference in max reached price before expiration and stock price) / stock price
	</small>
	<br><img src="cid:image5"><br>
	
	Plotting share of options which reached their strike price
	<br><img src="cid:image6"><br>
	
  </body>
""".format(hprob_config['optionType'], hprob_config['minStrikeIncrease'], hprob_config['maxStrikeIncrease']
		   , model_name, len(df), df['baseSymbol'].nunique()
		   , len(ReachedStrike), ReachedStrike['baseSymbol'].nunique()
		   , round(auc_roc,3), round(auc_pr,3) , round(brier_score,3)
		   , biggest_increase_df.to_html(), round(roi_highprob,3), round(roi_highprof,3))

if add_attachment:
	attachment = df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice',
					 'expirationDate', 'lastPrice', 'exportedAt',
					 'finalPrice', 'finalPriceDate', 'firstPrice', 'firstPriceDate',
					 'maxPrice', 'maxPriceDate', 'minPrice', 'minPriceDate',
					 'prob']].reset_index(drop=True)
else:
	attachment = None


password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()
sendRichEmail(sender='k.sends.python@gmail.com'
			  , receiver=emaillist
			  , password=password
			  , subject='Performance report expiry date {}'.format(last_friday)
			  , content=html_content
			  , inline_images=['scheduled_jobs/summary_content/scatter.png', 'scheduled_jobs/summary_content/CalibCurve.png',
							   'scheduled_jobs/summary_content/roc.png', 'scheduled_jobs/summary_content/scatter_profitability.png',
							   'scheduled_jobs/summary_content/scatter_maxProfitability.png', 'scheduled_jobs/summary_content/strikePerBins.png']
			  , attachment=attachment
			  )


