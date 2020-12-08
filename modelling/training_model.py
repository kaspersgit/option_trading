# Load packages
import pandas as pd
import numpy as np
import os
os.chdir('/home/pi/Documents/python_scripts/option_trading')
from sklearn.model_selection import train_test_split
from option_trading_nonprod.models.tree_based import *


#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1.csv')

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

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
# Train
# AdaBoost classifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

getwd = os.getcwd()
params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
AB_model = fit_AdaBoost(X_train.append(X_test), y_train.append(y_test), X_val, y_val, params, save_model = True, ab_path=getwd+'/trained_models/', name='AdaBoost_model_v1')

Adaprob = AB_model.predict_proba(X_val)[:,1]

# calibration
# calibrated classifier
calClf = CalibratedClassifierCV(AB_model, cv='prefit', method='sigmoid')
calClf.fit(X_val, y_val)
calClf.feature_names = X_train.columns

# Save model(S)
# Save calibration model
save_to = '{}{}.sav'.format(getwd+'/trained_models/', 'c_AB_v1')
pickle.dump(calClf, open(save_to, 'wb'))
print('Saved model to {}'.format(save_to))
