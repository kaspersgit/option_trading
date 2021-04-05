#%%
# Load in all csv files from source folder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import os
from option_trading_nonprod.aws import *
from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.process.stock_price_enriching import *

# Temp setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# read in enriched data
df_enr = pd.read_csv('data/marketbeat_yf_enr_1.csv')

#%%
# Add target variables
# What price we think we can buy it for?
# baseLastPrice ?
# nextBDopen ?
buyPrice = 'firstPrice'
# Add flag if stock raised X%
df_enr['high_plus50p'] = np.where(df_enr[buyPrice] * 1.5 <= df_enr['maxPrice'],1,0)
df_enr['high_plus30p'] = np.where(df_enr[buyPrice] * 1.3 <= df_enr['maxPrice'],1,0)
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr[buyPrice] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr[buyPrice] * 0.9 >= df_enr['minPrice'],1,0)

# remove rows not having buyPrice column
df_enr = df_enr[~df_enr[buyPrice].isna()]

#%%%
# add more columns
#variables
# extract from indicator column
df_enr['indicatorPresent'] = np.where(df_enr['indicators'].isnull(),0,1)
df_enr['upcomingEarning'] = np.where(df_enr['indicators'].str.contains('Upcoming Earnings', na=False),1,0)
df_enr['earningAnnounced'] = np.where(df_enr['indicators'].str.contains('Earnings Announcement', na=False),1,0)
df_enr['analystReport'] = np.where(df_enr['indicators'].str.contains('Analyst Report', na=False),1,0)
df_enr['heaveNewsReporting'] = np.where(df_enr['indicators'].str.contains('Heavy News Reporting', na=False),1,0)
df_enr['gapDown'] = np.where(df_enr['indicators'].str.contains('Gap Down', na=False),1,0)
df_enr['gapUp'] = np.where(df_enr['indicators'].str.contains('Gap Up', na=False),1,0)


df_enr['callStockVolume'] = df_enr['avgStockVolume'] / df_enr['avgOptionVolume']

# Get profit 
df_enr['revenue10p'] = np.where(df_enr['high_plus10p'] == 1, 1.1*df_enr[buyPrice], df_enr['finalPrice'])
# df_enr['revenue10p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue10p'])
df_enr['profit10p'] = df_enr['revenue10p'] - df_enr[buyPrice]
df_enr['profitPerc10p'] = df_enr['profit10p']/df_enr[buyPrice]

# for high of 30 percent 
df_enr['revenue30p'] = np.where(df_enr['high_plus30p'] == 1, 1.3*df_enr[buyPrice], df_enr['finalPrice'])
# df_enr['revenue30p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue30p'])
df_enr['profit30p'] = df_enr['revenue30p'] - df_enr[buyPrice]
df_enr['profitPerc30p'] = df_enr['profit30p']/df_enr[buyPrice]

# for high of 50 percent 
df_enr['revenue50p'] = np.where(df_enr['high_plus50p'] == 1, 1.5*df_enr[buyPrice], df_enr['finalPrice'])
#df_enr['revenue50p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue50p'])
df_enr['profit50p'] = df_enr['revenue50p'] - df_enr[buyPrice]
df_enr['profitPerc50p'] = df_enr['profit50p']/df_enr[buyPrice]

# weighted profit
df_enr['units'] = 1000/df_enr[buyPrice]
df_enr['weightedProfit30p'] = df_enr['units'] * df_enr['profit30p']
df_enr['weightedProfit50p'] = df_enr['units'] * df_enr['profit50p']

# feature selection
features = ['callOptionVolume', 'avgOptionVolume',
                'increaseRelative2Avg', 'avgStockVolume',
                'firstPrice', 'indicatorPresent',
                'upcomingEarning', 'earningAnnounced', 'heaveNewsReporting', 'analystReport',
            'gapUp','gapDown','callStockVolume']

########################
# Split in train, validation, test and out of time
# target  used ['high_plus10p','high_plus30p','high_plus50p']
target = 'high_plus50p'

# Split in train, validation, test and out of time
# Take most recent observations for out of time set (apprx last 5000 observations)
exportDateLast100 = df_enr.iloc[-100]['exportedAt']
df_oot = df_enr[df_enr['exportedAt'] >= exportDateLast100]
df_rest = df_enr.drop(df_oot.index, axis=0).reset_index(drop=True)

# test to split keeping exportedAt column always in same group
gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
gss.get_n_splits()

# split off test set
test_groupsplit = gss.split(df_rest, groups = df_rest['exportedAt'])
train_idx, test_idx = next(test_groupsplit)
df_rest2 = df_rest.loc[train_idx]
df_test = df_rest.loc[test_idx]

# split off validation set
df_rest2 = df_rest2.reset_index(drop=True)
val_groupsplit = gss.split(df_rest2, groups = df_rest2['exportedAt'])
train_idx, val_idx = next(val_groupsplit)
df_train = df_rest2.loc[train_idx]
df_val = df_rest2.loc[val_idx]

# clean unwanted columns for model training
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

X_oot = df_oot.drop(columns=[target])
y_oot = df_oot[target]

print("Train shape: {}\nValidation shape: {}\nTest shape: {}\nOut of time shape: {}".format(X_train.shape,X_val.shape,X_test.shape,X_oot.shape))
#####################
# Train and predict
# AdaBoost classifier
# either DEV(ELOPMENT) or PROD(UCTION)
# v1x0 trained on data with max expirationDate 2020-10-30
# v1x1 trained on data with max expirationDate 2020-12-18

train_type = 'DEV'
data_source = 'MB' # MB = MarketBeat, BC=BarChart
targetf = '50p'
version = 'v1x0'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test.to_csv("data/validation/{}_test_df.csv".format(data_source))
    df_oot.to_csv("data/validation/{}_oot_df.csv".format(data_source))
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test])
    y_fit = pd.concat([y_train, y_test])

getwd = os.getcwd()
params = {'n_estimators':1000, 'learning_rate': 0.01, 'max_depth':4, 'random_state':42, 'subsample':0.8}
GBC_model = fit_GBclf(X_train[features], y_train, X_val[features], y_val, params, save_model = True, gbc_path=getwd+'/trained_models/', name=data_source+'GB64_'+targetf+'_'+version)
# Calibrate pre trained model
Cal_GB_model = calibrate_model(GBC_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name=train_type+'_'+data_source+'_c_GB64_'+targetf+'_'+version)

###########
# Choose model
# Load model
getwd = os.getcwd()
with open(getwd+'/trained_models/DEV_MB_c_GB64_50p_v1x0.sav', 'rb') as file:
    gb_model = pickle.load(file)
with open(getwd+'/trained_models/DEV_c_AB64_v1x3.sav', 'rb') as file:
    ab_model = pickle.load(file)

model = gb_model

# Make predictions
prob = model.predict_proba(X_test[model.feature_names])[:,1]

pred_df = pd.DataFrame({'prob': prob, 'actual': y_test})
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5, 1, 0)

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC
from option_trading_nonprod.other.trading_strategies import *

# AUC
plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all data', type='roc')

# Confucion matrix
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

# Calibration plot
plotCalibrationCurve(pred_df['actual'], pred_df['prob'], title='all data', bins=10)

# Brier score
print(brier_score_loss(pred_df['actual'], pred_df['prob']))

# profitability
df_test['prob'] = prob
# to make the function work (as it is made for the options data
df_test['baseLastPrice'] = df_test['firstPrice']
df_test['strikePrice'] = df_test['firstPrice'] * 1.5
df_test['symbolType'] = 'Call'
roi, cost, revenue, profit = simpleTradingStrategy(df_test, actualCol = target, filterset={}, plot=True)

