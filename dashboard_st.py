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
import streamlit as st
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
        s3_profile = 'default'

    bucket = 'project-option-trading-output'
    key = 'enriched_data/barchart/expired_on_'
    df = load_from_s3(profile=s3_profile, bucket=bucket, key_prefix=key)

    print('Shape of raw imported data: {}'.format(df.shape))

    # enrich data within batches
    df = batch_enrich_df(df)
    print('Shape of batch enriched data: {}'.format(df.shape))

    return df


# Check if run locally (macbook) or not
if platform.system() == 'Darwin':
    modelname = 'DEV_c_GB64_v1x4'
else:
    os.chdir("/home/pi/Documents/python_scripts/option_trading")
    modelname = sys.argv[1]

# import data
# load in data from s3
df_all = load_data()

# Set title for sidebar
st.sidebar.markdown('## Filtering options')

# Set filter options
# set slider
today = date.today()
today_m100 = today - timedelta(days=100)
start_date = st.sidebar.date_input('Start date', today_m100)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Selected period of `%s` days' % ((end_date - start_date).days))
else:
    st.sidebar.error('Error: End date must fall after start date.')

# filter set on applicable rows
# only select Call option out of the money
with open('other_files/config_file.json') as json_file:
    config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']
included_options = config['included_options']

# Filter on basics (like days to expiration and contract type)
df = dfFilterOnGivenSetOptions(df_all, included_options)

print('Shape of filtered data: {}'.format(df.shape))

########################
# list available models
models = os.listdir(os.getcwd() + '/trained_models')
available_models = [i for i in models if i.startswith('PROD')]

# allow to choose model
# modelname = st.sidebar.selectbox('Select model:', available_models)

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

# basic performance
# accuracy (split per days to expiration)
# accuracy (split per strike price increase)
# brier score
brier_score = brier_score_loss(df['reachedStrikePrice'], df['prob'])
st.write('Brier score:', brier_score)

# Simulating trading strategies
print('Simulating simple trading strategies')

# High probability
roi_highprob, cost_highprob, revenue_highprob, profit_highprob = simpleTradingStrategy(df, actualCol='reachStrikePrice',
                                                                                       filterset=hprob_config,
                                                                                       plot=False)

# High profitability
roi_highprof, cost_highprof, revenue_highprof, profit_highprof = simpleTradingStrategy(df, actualCol='reachStrikePrice',
                                                                                       filterset=hprof_config,
                                                                                       plot=False)

####################
# PLOTS GENERATION #
####################

## pre plot variable and dftransformations
# Subset total df into ones reaching strike and those not
ReachedStrike = df[df['reachedStrikePrice'] == 1]
notReachedStrike = df[df['reachedStrikePrice'] == 0]

# Filter on options appearing in high probability, profitability or in neither of the two
high_prob_df = dfFilterOnGivenSetOptions(df, hprob_config)
high_prof_df = dfFilterOnGivenSetOptions(df, hprof_config)

# Get days to strike price as a share of possible options reaching it
df_days2strike = getDaysToStrikeAsShare(df)

print('Start creating plots')

##### General outcome
# scatter plot (strike percentage increase against predicted probability)
fig = PredictionVsStrikeIncreasePlotly(df, returnfig=True, savefig=False)
st.plotly_chart(fig)

# Create bar plots showing share of successes per strike price increase bucket
fig = GroupsPerformanceComparisonBarPlotly(df, high_prob_df, high_prof_df, returnfig=True, savefig=False)
st.plotly_chart(fig)

# Nr of days until options reach their strike price
fig = plotBarChartPlotly(df_days2strike, xcol='duration', ycol='strikeReachedShare',
                   titles={'title': 'Nr of days to reach strike price', 'xlabel': 'Days since extraction',
                           'ylabel': 'Share of active options'}, returnfig=True, savefig=False)
st.plotly_chart(fig)

##### Model performance
# calibration curve
fig = plotCalibrationCurve(df['reachedStrikePrice'], df['prob'], title='', bins=10, returnfig=True, savefig=False)
st.pyplot(fig)

##### Profitability
# Plot on expected profit when selling on strike (if reached)
fig = ExpvsActualProfitabilityScatter(df, high_prob_df, high_prof_df, actualCol='profitPerc', returnfig=True, savefig=False)
st.pyplot(fig)

# Plot on maximum profit if sold at max reached price
fig = ExpvsActualProfitabilityScatter(df, high_prob_df, high_prof_df, actualCol='maxProfitability', returnfig=True, savefig=False)
st.pyplot(fig)


# # precision threshold plot
# plotThresholdMetrics(df['prob'], df['reachedStrikePrice'], savefig=True,
#                      saveFileName='scheduled_jobs/summary_content/pr-threshold.png')
#
# print('Created and saved precision vs threshold plot')
#
# # AUC and similar
# auc_roc = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='roc', savefig=True,
#                        saveFileName='scheduled_jobs/summary_content/roc.png')
#
# print('Created and saved ROC plot')
#
# auc_pr = plotCurveAUC(df['prob'], df['reachedStrikePrice'], title='', type='pr', savefig=True,
#                       saveFileName='scheduled_jobs/summary_content/pr.png')
#
# print('Created and saved Precision Recall plot')