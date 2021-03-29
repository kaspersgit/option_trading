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
	model = 'DEV_c_GB64_v3x3'
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

# enriching based on platform with tehcnical indicators
if platform.system() == 'Darwin':
	# Get technical indicators
	# Get stock prices from 35 days before export date to calculate them
	df['exportedAt'] = pd.to_datetime(df['exportedAt'])
	df['start_date'] = df['exportedAt'] - timedelta(days=45)
	indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')
	indicators_df['exportedAt'] = pd.to_datetime(indicators_df['exportedAt'])

	# Put dfs together
	df = df.merge(indicators_df, on=['baseSymbol','exportedAt'])
	df.fillna(0,inplace=True)

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


# Basic summary
# Get top performing stocks (included/not included in email)
biggest_increase_df = df.sort_values('maxProfitability', ascending=False)[['baseSymbol','exportedAt','baseLastPrice','maxPrice','maxPriceDate','maxProfitability','prob']].drop_duplicates(subset=['baseSymbol']).head(10)
biggest_increase_df.reset_index(drop=True, inplace=True)

biggest_decrease_df = df.sort_values('maxProfitability', ascending=True)[['baseSymbol','exportedAt','baseLastPrice','maxPrice','maxPriceDate','maxProfitability','prob']].drop_duplicates(subset=['baseSymbol']).head(5)
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

## Extra df transformations for plots below
# Bucket strike price increase
bins = [-np.inf, 1.1, 1.15, 1.20, 1.25, 1.3, 1.4, 1.5, np.inf]
labels = ['5%-10%', '10%-15%', '15%-20%', '20%-25%', '25%-30%', '30%-40%', '40%-50%', '>50%']

df['strikePricePercBin'] = pd.cut(df['strikePricePerc'], bins=bins, labels=labels)

# Filter on options appearing in high probability, profitability or in neither of the two
high_prob_df = dfFilterOnGivenSet(df, hprob_config)
high_prof_df = dfFilterOnGivenSet(df, hprof_config)

####################################
# Create bar plots showing share of successes per strike price increase bucket
all_strikeIncreaseBin = df[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
all_binPercentage = all_strikeIncreaseBin.groupby(level=0).apply(lambda x:
												 100 * x / float(x.sum())).reset_index(drop=False)

hprob_strikeIncreaseBin = high_prob_df[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
hprob_binPercentage = hprob_strikeIncreaseBin.groupby(level=0).apply(lambda x:
																 100 * x / float(x.sum())).reset_index(drop=False)

hprof_strikeIncreaseBin = high_prof_df[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
hprof_binPercentage = hprof_strikeIncreaseBin.groupby(level=0).apply(lambda x:
																 100 * x / float(x.sum())).reset_index(drop=False)


fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# All contracts
axs[0].bar(all_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
axs[0].bar(all_binPercentage['strikePricePercBin'].unique(), all_binPercentage[all_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
axs[0].tick_params('x', labelrotation=45)
axs[0].title.set_text('All Calls')
axs[0].set_ylabel('Fraction reaching strike price')

# high probability contracts
axs[1].bar(hprob_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
axs[1].bar(hprob_binPercentage['strikePricePercBin'].unique(), hprob_binPercentage[hprob_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
axs[1].tick_params('x', labelrotation=45)
axs[1].title.set_text('High probability')
axs[1].set_xlabel("Strike price increase with respect to stock price")

# high probability contracts
axs[2].bar(hprof_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
axs[2].bar(hprof_binPercentage['strikePricePercBin'].unique(), hprof_binPercentage[hprof_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
axs[2].tick_params('x', labelrotation=45)
axs[2].title.set_text('High profitability')

fig.tight_layout(rect=[0,0,0.9,0.9])
fig.savefig("scheduled_jobs/summary_content/strikePerBins.png")

#################################### Unsure
# Create scatter plot (strike price progress vs predicted probability)
# rows not appearing in any of the email tables
not_email_df = df[(~df.index.isin(high_prob_df.index)) & (~df.index.isin(high_prof_df.index))]

fig = plt.figure()
# cm = plt.cm.get_cmap('Blues')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(not_email_df['profitPerc'], not_email_df['expPercIncrease'], s=7, color='r', alpha=0.7, label='Not in email')
ax.scatter(high_prob_df['profitPerc'], high_prob_df['expPercIncrease'], s=7, color='g', alpha=0.7, label='High probability')
ax.scatter(high_prof_df['profitPerc'], high_prof_df['expPercIncrease'], s=7, color='b', alpha=0.7, label='High profitability')
ax.legend(loc="upper left")
ax.set_xlabel('Actual profit')
ax.set_ylabel('Expected profit')
ax.set_title('Expected vs actual profitability')
plt.show()
fig.savefig("scheduled_jobs/summary_content/scatter_profitability.png")

print('Created and saved scatter plot (expected vs actual profitability')

fig = plt.figure()
# cm = plt.cm.get_cmap('Blues')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(not_email_df['maxProfitability'], not_email_df['expPercIncrease'], s=7, color='r', alpha=0.7, label='Not in email')
ax.scatter(high_prof_df['maxProfitability'], high_prof_df['expPercIncrease'], s=7, color='b', alpha=0.7, label='High profitability')
ax.scatter(high_prob_df['maxProfitability'], high_prob_df['expPercIncrease'], s=7, color='g', alpha=0.7, label='High probability')
ax.legend(loc="upper right")
ax.set_xlim(xmax=3)
ax.set_xlabel('Max profit')
ax.set_ylabel('Expected profit')
ax.set_title('Expected vs max profitability')
plt.show()
fig.savefig("scheduled_jobs/summary_content/scatter_maxProfitability.png")

print('Created and saved scatter plot (expected vs max profitability')
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


