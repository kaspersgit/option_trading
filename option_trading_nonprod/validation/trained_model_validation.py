import pandas as pd
import numpy as np


##### This script
# Predict test data
# Show performance on different subsets of data
# Show feature importance


###########
# Load data and model

# Load test data
pred_df = pd.read_csv("validation/test_df.csv")

# Load model
getwd = os.getcwd()
with open(getwd+'/trained_models/DEV_c_AB64_v1x0.sav', 'rb') as file:
    DEV_c_AB64_v1x0 = pickle.load(file)
with open(getwd+'/trained_models/DEV_c_AB64_v1x1.sav', 'rb') as file:
    DEV_c_AB64_v1x1 = pickle.load(file)
with open(getwd+'/trained_models/DEV_c_AB64_v1x2.sav', 'rb') as file:
    DEV_c_AB64_v1x2 = pickle.load(file)

model = DEV_c_AB64_v1x2



############
# Make predictions
prob = model.predict_proba(pred_df[model.feature_names])[:,1]
pred_df['prob'] = prob
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5,1,0)
pred_df.rename(columns={'reachedStrikePrice':'actual'}, inplace=True)

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC
from option_trading_nonprod.validation.calibration import *

# AUC
plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all test observations',type='roc')

# Confucion matrix
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

# Calibration plot
plotCalibrationCurve(pred_df['actual'], pred_df['prob'], title='all test observations', bins=10)

# Show perfromance for different segments
brackets = [{'lower':1.00, 'higher':1.02}
    ,{'lower':1.02, 'higher':1.05}
    ,{'lower':1.05, 'higher':1.1}
    , {'lower':1.1, 'higher':1.2}
    , {'lower':1.2, 'higher':9999}
            ]
for bracket in brackets:
    print('Strike price / stock price ratio between {} and {}'.format(bracket['lower'], bracket['higher']))

    title = 'ratio between {} and {}'.format(bracket['lower'], bracket['higher'])
    select_df = pred_df[(pred_df['priceDiffPerc'] >= bracket['lower']) & (pred_df['priceDiffPerc'] < bracket['higher'])]
    print('Nr observations {} out of {} ({})'.format(len(select_df),len(pred_df),round(len(select_df)/len(pred_df),2)))

    auc_roc = plotCurveAUC(select_df['prob'],select_df['actual'], title = title,type='roc')
    auc_pr = plotCurveAUC(select_df['prob'],select_df['actual'], title = title,type='pr')
    print('AUC ROC: {} \nAUC PR: {}'.format(round(auc_roc,3), round(auc_pr,3)))
    plotCalibrationCurve(select_df['actual'], select_df['prob'], title = title, bins=10)

# Find other in same batch
# df_all[(df_all['baseSymbol']=='MARA') & (pd.to_datetime(df_all['exportedAt']).dt.strftime('%Y-%m-%d') == '2020-12-23')].iloc[0]

###########
# Feature importance
from sklearn.inspection import permutation_importance
permutation_imp = permutation_importance(model, pred_df[model.feature_names], pred_df['actual'])
feat_permutation_imp = pd.DataFrame({'feature': model.feature_names, 'importance': permutation_imp.importances_mean}).sort_values('importance', ascending=False).reset_index(drop=True)
feat_impurity_imp = pd.DataFrame({'feature': model.feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)

# Scrablle
sdf =  pred_df[(pred_df['priceDiffPerc'] >= 1.1) & (pred_df['priceDiffPerc'] < 1.2) & (pred_df['prob'] > 0.8) & (pred_df['prob'] < 0.9)]