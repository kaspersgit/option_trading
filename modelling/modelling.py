# Load packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.validation.calibration import *

#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1.csv')

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

# TODO include this in preprocesseing
# clean unwanted columns
df = df.drop(columns=['Unnamed: 0','baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'])

########################
# Split in train and test
X = df.drop(columns='reachedStrikePrice')
y = df['reachedStrikePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#####################
# Train and predict

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Random Forest classifier model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)

# AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

clf = AdaBoostClassifier(n_estimators=1000, learning_rate=0.5, random_state=42)
clf.feature_names=X_train.columns
clf.fit(X_train,y_train)

getwd = os.getcwd()
params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
AB_model = fit_AdaBoost(X_train.append(X_test), y_train.append(y_test), X_val, y_val, params, save_model = True, ab_path=getwd+'/trained_models/', name='AdaBoost_model_v1')

params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
GBC_model = fit_GBclf(X_train, y_train, X_val, y_val, params, save_model = True, gbc_path=getwd+'/trained_models/', name='GBclf_v1')

Adaprob = clf.predict_proba(X_val)[:,1]
GBprob = GBC_model.predict_proba(X_val)[:,1]

# calibration
# isotonic regression
iso_reg = IsotonicRegression().fit(Adaprob, y_val)
# calibrated classifier
calClf = CalibratedClassifierCV(GBC_model, cv='prefit', method='sigmoid')
calClf.fit(X_val, y_val)
calClf.feature_names = X_train.columns

# test Dataset
rawprob = clf.predict_proba(X_test)[:,1]
prob_iso_reg = iso_reg.predict(rawprob)
prob = calClf.predict_proba(X_test)[:,1]

plot_calibration_curve(AdaBoostClassifier(),X_train,y_train,X_test,y_test,'AdaBoost',1)
plot_calibration_curve(GradientBoostingClassifier(n_estimators=500),X_train,y_train,X_test,y_test,'GradientBoostingClf',1)

# Catboost (not working on raspberry pi (32bit))
# params = {'iterations':300}
#
# getwd = os.getcwd()
# cb_model = fit_cb(X_train, y_train, X_val, y_val, params, save_model = True, cb_path=getwd+'/trained_models/', name='cb_model_v1')


# Save model(S)
# Save calibration model
save_to = '{}{}.sav'.format(getwd+'/trained_models/', 'c_GB_v1')
pickle.dump(calClf, open(save_to, 'wb'))
print('Saved model to {}'.format(save_to))


###########
# Choose model
# Load model
getwd = os.getcwd()
with open(getwd+'/trained_models/AB_v1.sav', 'rb') as file:
    model = pickle.load(file)
with open(getwd+'/trained_models/calibrated_AdaBoost_model_v1.sav', 'rb') as file:
    calib_model = pickle.load(file)

model = GBC_model

# Make predictions
prob = model.predict_proba(X_test)[:,1]


pred_df = pd.DataFrame({'prob':prob, 'actual':y_test})
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5,1,0)

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC

plotCurveAUC(pred_df['prob'],pred_df['actual'],type='roc')
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

#####################
# Visualize

######################
# Test out predictions
# profitability
df_test = df_all.loc[pred_df.index,:]
df_test['prob'] =  pred_df['prob']
df_test['priceDiffPerc'] = df_test['strikePrice'] / df_test['baseLastPrice']
df_test['maxProfit'] = df_test['maxPrice'] - df_test['baseLastPrice']
df_test['aimedProfit'] = np.where(df_test['maxPrice'] >= df_test['strikePrice'],df_test['strikePrice'], df_test['finalPrice']) - df_test['baseLastPrice']
# Select based on parameters
# Subsetting the predictions
threshold = 0.7
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 20
minStrikeIncrease = 1.05

df_test = df_test[(df_test['prob'] > threshold) &
    (df_test['symbolType']=='Call') &
    (df_test['daysToExpiration'] < maxDaysToExp) &
    (df_test['priceDiffPerc'] > minStrikeIncrease) &
    (df_test['daysToExpiration'] > minDaysToExp) &
    (df_test['baseLastPrice'] < maxBasePrice)]

df_test.describe()


######################
# Tune hyperparameters