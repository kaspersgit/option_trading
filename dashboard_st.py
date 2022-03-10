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
from dateutil.relativedelta import relativedelta, FR
import os, sys, platform
import pickle
import json
import streamlit as st
import hashlib
from datetime import date, timedelta

from option_trading_nonprod.aws import *
from option_trading_nonprod.other.trading_strategies import *
from option_trading_nonprod.other.specific_plots import *
from option_trading_nonprod.utilities.email import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.validation.classification import *
from option_trading_nonprod.validation.plotting import *
from option_trading_nonprod.process.stock_price_enriching import *
from option_trading_nonprod.process.simple_enriching import *

@st.cache
def load_data():
	if platform.system() == 'Darwin':
		s3_profile = 'mrOption'
	else:
		s3_profile = 'streamlit'

	bucket = 'project-option-trading-output'
	key = 'enriched_data/barchart/expired_on_'

	df_all = load_from_s3(profile=s3_profile, bucket=bucket, key_prefix=key)

	print('Shape of raw imported data: {}'.format(df_all.shape))

	# enrich data within batches
	df = batch_enrich_df(df_all)
	print('Shape of batch enriched data: {}'.format(df.shape))

	df = addTargets(df)

	df = cleanDF(df)

	return df.reset_index(drop=True)

# password implementation
password = st.sidebar.text_input('Type in password')
if os.environ['PASSWORD'] == password:
    st.sidebar.write('Password is correct')
else:
    st.sidebar.write('Password is wrong')

# Set title for sidebar
st.sidebar.markdown('## Filtering options')

# filter set on applicable rows
# only select Call option out of the money
with open('other_files/config_file.json') as json_file:
    config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']
included_options = config['included_options']

########################
## SELECTION OF MODEL ##
########################

# list available models
models = os.listdir(os.getcwd() + '/trained_models')
available_models = [i for i in models if ('64' in i)]

# allow to choose model
modelname = st.sidebar.selectbox('Select model:', available_models, index=available_models.index('DEV_c_GB64_v1x4.sav'))
modelname = modelname.split('.')[0]

if 'GB' in modelname:
    model_expl_url = 'https://en.wikipedia.org/wiki/Gradient_boosting'
    model_descr = 'Gradient Boosting'
elif 'CB' in modelname:
    model_expl_url = 'https://en.wikipedia.org/wiki/Catboost'
    model_descr = 'CatBoost'
elif 'EBM' in modelname:
    model_expl_url = 'https://interpret.ml/docs/ebm.html'
    model_descr = 'Explainable Boosting Machine'

# Set filter options
# for expiration date
today = date.today()
today_m100 = today - timedelta(days=10)
start_date = st.sidebar.date_input('Start date', today_m100)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Selected period of `%s` days' % ((end_date - start_date).days))
else:
    st.sidebar.error('Error: End date must fall after start date.')

# for other options (as in the config jsons
min_price_range = st.sidebar.number_input('Minimum stock price at start', min_value=0, max_value=500, value=0)
max_price_range = st.sidebar.number_input('Maximum stock price at start', min_value=0, max_value=500, value=200)
min_extr2exp_days = st.sidebar.number_input('Minimum nr of days until expiration', min_value=0, max_value=100, value=5)
max_extr2exp_days = st.sidebar.number_input('Maximum nr of days until expiration', min_value=0, max_value=100, value=20)
min_strikprice_increase = st.sidebar.number_input('Minimum strike price increase ratio', min_value=1.0, max_value=20.0, value=1.05)
max_strikprice_increase = st.sidebar.number_input('Maximum strike price increase ratio', min_value=1.0, max_value=20.0, value=10.0)
threshold_range = st.sidebar.number_input('Minimal probability', min_value=0.0, max_value=1.0, value=0.0)

# import data
# load in data from s3
df_all = load_data()
df = df_all.copy()

# update included options filter
included_options['maxBasePrice'] = max_price_range
included_options['minDaysToExp'] = min_extr2exp_days
included_options['maxDaysToExp'] = max_extr2exp_days
included_options['minStrikeIncrease'] = min_strikprice_increase
included_options['maxStrikeIncrease'] = max_strikprice_increase
included_options['minThreshold'] = threshold_range
included_options['maxThreshold'] = 1

########################
# Import model and score
file_path = os.getcwd() + '/trained_models/' + modelname + '.sav'
with open(file_path, 'rb') as file:
    model = pickle.load(file)
features = model.feature_names
# Below will filter out the interaction feature names
features = [f for f in features if ' x ' not in f]
prob = model.predict_proba(df[features])[:, 1]

print('Loaded model and scored options')

####### Adding columns and variables #############
# Add prediction variable
df['prob'] = prob

# set threshold
threshold = 0.5
df['pred'] = np.where(df['prob'] > 0.5,1,0)

# Filter on basics (like days to expiration and contract type)
df = dfFilterOnGivenSetOptions(df, included_options)

print('Shape of filtered data: {}'.format(df.shape))

df = simpleEnriching(df)

###############################
######  apply custom filter
df = df.loc[(df['expirationDate']>=str(start_date))
                & (df['expirationDate']<=str(end_date))]

################################################
# Basic summary
# Get top performing stocks (included/not included in email)
biggest_increase_df = df.sort_values('maxProfitability', ascending=False)[
    ['baseSymbol', 'exportedAt', 'baseLastPrice', 'maxPrice', 'maxPriceDate', 'maxProfitability']].drop_duplicates(
    subset=['baseSymbol']).head(5)
biggest_increase_df.reset_index(drop=True, inplace=True)

biggest_decrease_df = df.sort_values('maxProfitability', ascending=True)[
    ['baseSymbol', 'exportedAt', 'baseLastPrice', 'maxPrice', 'maxPriceDate', 'maxProfitability']].drop_duplicates(
    subset=['baseSymbol']).head(5)
biggest_decrease_df.reset_index(drop=True, inplace=True)


# High probability
roi_highprob, cost_highprob, revenue_highprob, profit_highprob = simpleTradingStrategy(df, actualCol='reachStrikePrice',
                                                                                       filterset=hprob_config,
                                                                                       plot=False)

# High profitability
roi_highprof, cost_highprof, revenue_highprof, profit_highprof = simpleTradingStrategy(df, actualCol='reachStrikePrice',
                                                                                       filterset=hprof_config,
                                                                                       plot=False)

# Filter on options appearing in high probability, profitability or in neither of the two
high_prob_df = dfFilterOnGivenSetOptions(df, hprob_config)
high_prof_df = dfFilterOnGivenSetOptions(df, hprof_config)

# Get days to strike price as a share of possible options reaching it
df_days2strike = getDaysToStrikeAsShare(df)

#################
# Create dashboard content variables
################

# AUC for ROC and PR
fig_roc, auc_roc = plotCurveAUCPlotly(df['prob'], df['reachedStrikePrice'], title='', type='roc', returnfig=True,
                                  savefig=False,
                                  saveFileName='scheduled_jobs/summary_content/roc.png')


fig_pr, auc_pr = plotCurveAUCPlotly(df['prob'], df['reachedStrikePrice'], title='', type='pr', returnfig=True,
                                    savefig=False,
                                    saveFileName='scheduled_jobs/summary_content/pr.png')

brier_score = brier_score_loss(df['reachedStrikePrice'], df['prob'])

############################
# Start dashboard creation #
############################

st.markdown('# Option trading model summary')

st.markdown('## Data information')

st.write(f'Total number of options (unique tickers): {len(df)}({df.baseSymbol.nunique()})')
st.write(f'Options reaching strike (unique tickers): {df.reachedStrikePrice.sum()}({df[df.reachedStrikePrice == 1].baseSymbol.nunique()})')
st.write(f'Share of options reaching strike: {df.reachedStrikePrice.sum() / len(df):.2f}')

st.markdown('## Model performance metrics')

st.write(f'Area Under Curve (AUC) of ROC: {auc_roc:.2f}')
st.write(f'AUC of Precision Recall: \t {auc_pr:.2f}')
st.write(f'Brier loss score: {brier_score:.2f}')

with st.expander("See model details and performance metrics"):
    st.write(f"""
        For predicting the chance an option will reach its strike price we used the following model \n
        Model name: {modelname} \n
        Model type: [{model_descr}]({model_expl_url}) \n
    """)

    # top 10 features of model
    model_feature_importance = pd.DataFrame(
        {'Feature': model.feature_names, 'Importance': model.feature_importances_}).sort_values('Importance',
                                                                                                ascending=False).reset_index(
        drop=True)
    st.table(model_feature_importance.head(10))

    st.plotly_chart(fig_roc)

    st.plotly_chart(fig_pr)

# Showing some model info
# model_details_df = pd.DataFrame({'Key':['model name', 'nr features', 'train size', 'calibration size']
#                                  , 'Value':[modelname, len(model.feature_names), model.train_data_shape[0]
#         , model.calibration_data_shape[0]]})
# st.table(model_details_df)



####################
# PLOTS GENERATION #
####################

print('Start creating plots')

##### General outcome
# scatter plot (strike percentage increase against predicted probability)
fig = PredictionVsStrikeIncreasePlotly(df, returnfig=True, savefig=False)
st.plotly_chart(fig)

# Create bar plots showing share of successes per strike price increase bucket
fig = GroupsPerformanceComparisonBarPlotly(df, high_prob_df, high_prof_df, returnfig=True, savefig=False)
st.plotly_chart(fig)

##### Stock price behaviour
# Nr of days until options reach their strike price
fig = plotBarChartPlotly(df_days2strike, xcol='duration', ycol='strikeReachedShare',
                   titles={'title': 'Nr of days to reach strike price', 'xlabel': 'Days since extraction',
                           'ylabel': 'Share of active options'}, returnfig=True, savefig=False)
st.plotly_chart(fig)

# Lowest price reached as % of starting price
fig = plotLowestPriceReachedPlotly(df, returnfig=True)
st.plotly_chart(fig)

##### Model performance
# calibration curve
fig = plotCalibrationCurvePlotly(df['reachedStrikePrice'], df['prob'], title='', bins=10, returnfig=True, savefig=False)
st.plotly_chart(fig)

# # precision threshold plot
fig = plotThresholdMetricsPlotly(df['prob'], df['reachedStrikePrice'], returnfig=True, savefig=False,
                     saveFileName='scheduled_jobs/summary_content/pr-threshold.png')
st.plotly_chart(fig)

##### Profitability
# Biggest increases
st.table(biggest_increase_df)

# Plot on expected profit when selling on strike (if reached)
fig = ExpvsActualProfitabilityScatter(df, high_prob_df, high_prof_df, actualCol='profitPerc', returnfig=True, savefig=False)
st.pyplot(fig)

# Plot on maximum profit if sold at max reached price
fig = ExpvsActualProfitabilityScatter(df, high_prob_df, high_prof_df, actualCol='maxProfitability', returnfig=True, savefig=False)
st.pyplot(fig)




