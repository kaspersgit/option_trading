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

from option_trading_nonprod.aws import *
from option_trading_nonprod.other.trading_strategies import *
from option_trading_nonprod.other.specific_plots import *
from option_trading_nonprod.utilities.email import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.validation.classification import *
from option_trading_nonprod.validation.plotting import *
from option_trading_nonprod.process.stock_price_enriching import *

# Check if run locally (macbook) or not
if platform.system() == 'Darwin':
	modelname = 'DEV_c_GB64_v1x4'
else:
	os.chdir("/home/pi/Documents/python_scripts/option_trading")
	modelname = sys.argv[1]

# Get supplied system arguments
# mode (development or production)
if len(sys.argv) >= 4:
	date = pd.to_datetime(sys.argv[3])
	last_friday = (date + relativedelta(weekday=FR(-1)))
else:
	last_friday = (datetime.today() + relativedelta(weekday=FR(-1)))

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

else:
	print('Script can be run from command line as <script> <model> <env prod or dev> <add attachment true/false> <date (optional)>')
	mode = 'DEVELOPMENT'
	add_attachment = False
	with open('/home/pi/Documents/trusted/option_email_list_dev.txt') as f:
		emaillist = f.read().splitlines()

# print status of variables
print('Mode: {}'.format(mode))
print('Emaillist: {}'.format(emaillist))

# Get model which should be used
modelname = modelname.split('.')[0]

# import data
if platform.system() == 'Darwin':
	s3_profile = 'mrOption'
else:
	s3_profile = 'default'

# Set bucket
bucket = 'project-option-trading-output'
# Get all dates from last friday until saturday previous
numdays = 7
date_list = [(last_friday - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(numdays)]

# check which what expiration date(s) are present
s3_client = connect_to_s3(s3_profile, type="client")

# For all dates try search for filename containing it
for d in date_list:
	possible_key = 'enriched_data/barchart/expired_on_{}.csv'.format(d)
	exist, key = get_s3_key(s3_client, bucket, possible_key)
	if exist:
		break
	if (d==date_list[-1]) & (not exist):
		print("No expiration date found")

# print status of variables
print('Model : {}'.format(modelname))
print('Last expiry date: {}'.format(d))
print('Last expiry weekday: {}'.format(datetime.strptime(d,'%Y-%m-%d').strftime('%A')))
print('Source bucket: {}'.format(bucket))
print('Source key: {}'.format(key))

# load in data from s3
df = load_from_s3(profile=s3_profile, bucket=bucket, key_prefix=key)


print('Shape of raw imported data: {}'.format(df.shape))

# enrich data within batches
df = batch_enrich_df(df)
print('Shape of batch enriched data: {}'.format(df.shape))

# filter set on applicable rows
# only select Call option out of the money
with open('other_files/config_file.json') as json_file:
	config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']
included_options = config['included_options']

# Filter on basics (like days to expiration and contract type)
df = dfFilterOnGivenSetOptions(df, included_options)

print('Shape of filtered data: {}'.format(df.shape))

# enriching based on platform with tehcnical indicators
if 'v3' in modelname:
	# Get technical indicators
	# Get stock prices from 35 days before export date to calculate them
	df['exportedAt'] = pd.to_datetime(df['exportedAt'])
	df['start_date'] = df['exportedAt'] - timedelta(days=45)
	indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')
	indicators_df['exportedAt'] = pd.to_datetime(indicators_df['exportedAt'])

	# Put dfs together
	df = df.merge(indicators_df, on=['baseSymbol','exportedAt'])
	df.fillna(0,inplace=True)

########################
# Import model and score
file_path = os.getcwd() + '/trained_models/' + modelname + '.sav'
with open(file_path, 'rb') as file:
	model = pickle.load(file)
features = model.feature_names
prob = model.predict_proba(df[features])[:, 1]

print('Loaded model and scored options')

####### Adding columns and variables #############
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
# Basic summary
# Get top performing stocks (included/not included in email)
biggest_increase_df = df.sort_values('maxProfitability', ascending=False)[['baseSymbol','exportedAt','baseLastPrice','maxPrice','maxPriceDate','maxProfitability']].drop_duplicates(subset=['baseSymbol']).head(5)
biggest_increase_df.reset_index(drop=True, inplace=True)

biggest_decrease_df = df.sort_values('maxProfitability', ascending=True)[['baseSymbol','exportedAt','baseLastPrice','maxPrice','maxPriceDate','maxProfitability']].drop_duplicates(subset=['baseSymbol']).head(5)
biggest_decrease_df.reset_index(drop=True, inplace=True)

# basic performance
# accuracy (split per days to expiration)
# accuracy (split per strike price increase)
# brier score
brier_score = brier_score_loss(df['reachedStrikePrice'], df['prob'])

# Simulating trading strategies
print('Simulating simple trading strategies')

# High probability
roi_highprob, cost_highprob, revenue_highprob, profit_highprob = simpleTradingStrategy(df, actualCol = 'reachStrikePrice', filterset=hprob_config, plot=False)

# High profitability
roi_highprof, cost_highprof, revenue_highprof, profit_highprof = simpleTradingStrategy(df, actualCol = 'reachStrikePrice', filterset=hprof_config, plot=False)



####################
# PLOTS GENERATION #
####################

## pre plot variable and dftransformations
# Subset total df into ones reaching strike and those not
ReachedStrike = df[df['reachedStrikePrice'] == 1]
notReachedStrike = df[df['reachedStrikePrice'] == 0]

# Bucket strike price increase
bins = [-np.inf, 1.1, 1.15, 1.20, 1.25, 1.3, 1.4, 1.5, np.inf]
labels = ['5%-10%', '10%-15%', '15%-20%', '20%-25%', '25%-30%', '30%-40%', '40%-50%', '>50%']

df['strikePricePercBin'] = pd.cut(df['strikePricePerc'], bins=bins, labels=labels)

# Filter on options appearing in high probability, profitability or in neither of the two
high_prob_df = dfFilterOnGivenSetOptions(df, hprob_config)
high_prof_df = dfFilterOnGivenSetOptions(df, hprof_config)

# Get days to strike price as a share of possible options reaching it
df_days2strike = getDaysToStrikeAsShare(df)

print('Start creating plots')

##### General outcome
# scatter plot (strike percentage increase against predicted probability)
PredictionVsStrikeIncrease(df, ReachedStrike, notReachedStrike, savefig=True, saveFileName="scheduled_jobs/summary_content/scatter.png")

print('Created and saved scatter plot (percentage increase vs predicted probability')

# Create bar plots showing share of successes per strike price increase bucket
GroupsPerformanceComparisonBar(df, high_prob_df, high_prof_df, savefig=True, saveFileName="scheduled_jobs/summary_content/strikePerBins.png")

# Nr of days until options reach their strike price
plotHistogramPlotly(df, col='duration', titles = {'title':'Nr of days to reach strike price', 'xlabel':'Days since extraction'}, savefig=True, saveFileName="scheduled_jobs/summary_content/daysToReachStrike.png")
plotBarChartPlotly(df, xcol='duration', ycol='strikeReachedShare', titles = {'title':'Nr of days to reach strike price', 'xlabel':'Days since extraction', 'ylabel':'Share of active options'}, savefig=True, saveFileName='scheduled_jobs/summary_content/daysToReachStrike.png')

##### Profitability
# Plot on expected profit when selling on strike (if reached)
ExpvsActualProfitabilityScatter(df,high_prob_df, high_prof_df, actualCol='profitPerc',  savefig=True, saveFileName="scheduled_jobs/summary_content/scatter_profitability.png")
# Plot on maximum profit if sold at max reached price
ExpvsActualProfitabilityScatter(df,high_prob_df, high_prof_df, actualCol='maxProfitability',  savefig=True, saveFileName="scheduled_jobs/summary_content/scatter_maxProfitability.png")

##### Model performance
# calibration curve
plotCalibrationCurve(df['reachedStrikePrice'], df['prob'], title='', bins=10, savefig=True,
					 saveFileName='scheduled_jobs/summary_content/CalibCurve.png')

print('Created and saved calibration plot')

# precision threshold plot
plotThresholdMetrics(df['prob'], df['reachedStrikePrice'], savefig=True,
					 saveFileName='scheduled_jobs/summary_content/pr-threshold.png')

print('Created and saved precision vs threshold plot')

# AUC and similar
auc_roc = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='roc', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/roc.png')

print('Created and saved ROC plot')

auc_pr = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='pr', savefig=True,
					   saveFileName='scheduled_jobs/summary_content/pr.png')

print('Created and saved Precision Recall plot')

###########################
# Composing email #
###########################

print('Composing email...')
# Send email
# recipient
# lay out and content
# attachment (the csv file)
html_content = f"""
<html>
  <head></head>
  <body>
	A summary of the call options expired last {datetime.strptime(d,'%Y-%m-%d').strftime('%A')} and the model's performance.
	<br>
	Showing only options of type: {hprob_config['optionType']}
	<br>
	With minimal price increase between: {hprob_config['minStrikeIncrease']} and {hprob_config['maxStrikeIncrease']}
	<br>
	Days to expiration between: {hprob_config['minDaysToExp']} and {hprob_config['maxDaysToExp']}
	<br>
	Maximum stock price of {hprob_config['maxBasePrice']}$
	<br>
	Model used: {modelname}
	<br><br>
	Total number of options (unique tickers): {len(df)} ({df['baseSymbol'].nunique()})
	<br>
	Options reaching strike (unique tickers): {len(ReachedStrike)} ({ReachedStrike['baseSymbol'].nunique()})
	<br>
	Share of options reaching strike: {round(len(ReachedStrike)/len(df),3)}
	<br><br>
	<hr>
	<h3>Model performance metrics</h3>
	
	Area Under Curve of ROC: 	{round(auc_roc,3)}
	<br>
	AUC of Precision Recall: 	{round(auc_pr,3)}
	<br>
	Brier loss score: 			{round(brier_score,3)}
	<br><br>
	<hr>

	<h3> Graphs for visual interpretation</h3>
	Plotting all options based on their profitability and probability
	<br><img src="cid:image1"><br><br>

	Plot of the calibration curve to see if the probabilities made sense
	<br><img src="cid:image2"><br><br>

	Plot of the share of options which reached their strike price
	<br><img src="cid:image3"><br><br>
	
	Histogram on the nr of days until strike price is reached
	<br><img src="cid:image4"><br><br>
	
	Plot of the precision and recall versus the threshold.
	<br>
	<small>
	Precision is the share of predicted options actually reaching the strike price
	</small>
	<br><img src="cid:image5"><br><br>
	
	Plotting the ROC, which gives an idea on how well the model performs
	<br><img src="cid:image6"><br><br>

	<br><br>
	<hr>
	<h3>Most profitable stocks</h3>
	<br>
	{biggest_increase_df.to_html()}
	<br><br>
	<hr>
	<h3>Profitability by trading stocks</h3>
	<br>
	<small>
	Buying the stock based on email, selling the stock when strike price is reached or on expiration date
	</small>
	<br>
	Return on investment (ROI):
	<br>
	High probability: {round(roi_highprob,3)}
	<br>
	High profitability: {round(roi_highprof,3)}
	<br><br>
	<hr>
	
	<h3>Plotting profitability</h3>
	<br>
	Expected profitability vs actual profitability
	<small> 
	Expected: (strike price increase * predicted probability) / stock price
	<br>
	Actual:	(strike price (if reached) or stock price on expiration minus stock price) / stock price
	</small>
	<br><img src="cid:image7"><br>
	
	Plotting expected profitability vs max profitability
	<small> 
	Expected: (strike price increase * predicted probability) / initial stock price
	<br>
	Max:	(max price before expiration minus stock price) / stock price
	</small>
	<br><img src="cid:image8"><br>
	

	
  </body>
"""

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
			  , subject='Performance report expiry date {}'.format(d)
			  , content=html_content
			  , inline_images=['scheduled_jobs/summary_content/scatter.png', 'scheduled_jobs/summary_content/CalibCurve.png',
							   'scheduled_jobs/summary_content/strikePerBins.png', "scheduled_jobs/summary_content/daysToReachStrike.png",
							   'scheduled_jobs/summary_content/pr-threshold.png' ,
							   'scheduled_jobs/summary_content/roc.png',
							   'scheduled_jobs/summary_content/scatter_profitability.png', 'scheduled_jobs/summary_content/scatter_maxProfitability.png']
			  , attachment=attachment
)


