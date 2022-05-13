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
def load_data(date=date.today().strftime('%Y-%m-%d')):
    if platform.system() == 'Darwin':
        s3_profile = 'mrOption'
    else:
        s3_profile = 'streamlit'

    bucket = 'project-option-trading'
    key = f'raw_data/barchart/barchart_unusual_activity_{date}.csv'

    df_all = load_from_s3(profile=s3_profile, bucket=bucket, key_prefix=key)

    print('Shape of raw imported data: {}'.format(df_all.shape))

    # enrich data within batches
    df = batch_enrich_df(df_all)
    print('Shape of batch enriched data: {}'.format(df.shape))

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

st.markdown('# Option trading model daily prediction')

st.markdown('## Model selection')

# list available models
models = os.listdir(os.getcwd() + '/trained_models')
available_models = [i for i in models if ('64' in i)]

# allow to choose model
modelname = st.selectbox('Select model:', available_models, index=available_models.index('DEV_c_GB64_v1x4.sav'))
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
check_date = st.sidebar.date_input('Date', today)

if check_date <= today:
    st.sidebar.success('Selected date')
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
df_all = load_data(check_date.strftime('%Y-%m-%d'))
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
df['model'] = modelname

# Adding some additional columns
df['predictionTimeStamp'] = df['exportedAt']
df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
df['inTheMoney'] = np.where(df['baseLastPrice'] >= df['strikePrice'],1,0)
df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','inTheMoney'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'
        ]).agg({'baseSymbol':'count', 'strikePrice':'mean'
        }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
        }).reset_index()
df = pd.merge(df,df_symbol, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'])





# Filter on basics (like days to expiration and contract type)
df = dfFilterOnGivenSetOptions(df, included_options)

# formatting
df['baseLastPrice'] = round(df['baseLastPrice'],2)
df['priceDiffPerc'] = round(df['priceDiffPerc'],2)
df.rename(columns={'baseSymbol': 'ticker',
				   'baseLastPrice': 'stockPrice',
				   'priceDiffPerc': 'increase'}, inplace=True)

print('Shape of filtered data: {}'.format(df.shape))

##############################
## make high probability and high profitablity dataframes
high_prob = df[(df['prob'] > hprob_config['minThreshold']) &
    (df['symbolType']=='Call') &
    (df['daysToExpiration'] < hprob_config['maxDaysToExp']) &
    (df['increase'] > hprob_config['minStrikeIncrease']) &
    (df['daysToExpiration'] > hprob_config['minDaysToExp']) &
    (df['stockPrice'] < hprob_config['maxBasePrice'])].copy()
high_prob = high_prob[['ticker', 'predictionTimeStamp', 'expirationDate', 'stockPrice', 'strikePrice', 'increase', 'prob', 'model']]
high_prob = high_prob.sort_values('increase').reset_index(drop=True)

print('High probability table size: {}'.format(len(high_prob)))

# Subsetting the predictions for highly profitable stocks
high_prof = df[(df['prob'] > hprof_config['minThreshold']) &
    (df['symbolType']=='Call') &
    (df['daysToExpiration'] < hprof_config['maxDaysToExp']) &
    (df['increase'] > hprof_config['minStrikeIncrease']) &
    (df['daysToExpiration'] > hprof_config['minDaysToExp']) &
    (df['stockPrice'] < hprof_config['maxBasePrice'])].copy()
high_prof = high_prof[['ticker', 'predictionTimeStamp', 'expirationDate', 'stockPrice', 'strikePrice', 'increase', 'prob', 'model']]
high_prof = high_prof.sort_values('increase').reset_index(drop=True)

print('High profitability table size: {}'.format(len(high_prof)))

################################################
# Basic summary

############################
# Continue dashboard creation #
############################
st.markdown('## Data information')

st.write(f"Showing options extracted on {check_date.strftime('%Y-%m-%d')}")
st.write(f'Total number of options (unique tickers): {len(df)}({df.ticker.nunique()})')

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

    # Showing some model info
    if hasattr(model,'feature_names') & hasattr(model,'train_data_shape') & hasattr(model,'calibration_data_shape'):
        model_details_df = pd.DataFrame({'Key':['model name', 'nr features', 'train size', 'calibration size']
                                         , 'Value':[modelname, str(len(model.feature_names)), str(model.train_data_shape[0])
                , str(model.calibration_data_shape[0])]})
        st.table(model_details_df)


####################
# DF GENERATION #
####################

##### Profitability
# Biggest increases
st.markdown('## High probability options')
st.table(high_prob)

st.markdown('## High profitability options')
st.table(high_prof)
