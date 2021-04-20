import pandas as pd
import numpy as np
import os
import pickle

from option_trading_nonprod.other.trading_strategies import simpleTradingStrategy
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC
from option_trading_nonprod.validation.calibration import *


##### This script
# Predict test data
# Show performance on different subsets of data
# Show feature importance

def modelPerformanceReport(model, dataset, ext_plots=False):
    if hasattr(model, 'base_estimator'):
        model_name = model.base_estimator
    else:
        model_name = model
    print("Model:\n{}".format(model_name))

    pred_df = dataset.copy()
    pred_df.fillna(0, inplace=True)
    prob = model.predict_proba(pred_df[model.feature_names])[:,1]
    pred_df['prob'] = prob
    pred_df['pred'] = np.where(pred_df['prob'] >= 0.5,1,0)
    pred_df.rename(columns={'reachedStrikePrice':'actual'}, inplace=True)

    # Create column to segment on
    pred_df['priceDiffPerc'] = pred_df['strikePrice'] / pred_df['baseLastPrice']
    pred_df = pred_df[pred_df['priceDiffPerc'] >= 1.05]

    ### Scatter plots
    # Plot market performance priceDiffPerc against extractedAt day
    # scatter plot (strike percentage increase against predicted probability)
    ReachedStrike = pred_df[pred_df['actual'] == 1]
    notReachedStrike = pred_df[pred_df['actual'] == 0]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(notReachedStrike['exportedAt'], notReachedStrike['priceDiffPerc'], s = 7, color='r', alpha=0.7,
                label='Not reached strike')
    ax1.scatter(ReachedStrike['exportedAt'], ReachedStrike['priceDiffPerc'], s = 7, color='g', alpha=0.7, label='Did reach strike')
    plt.title(model.version)
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.show()

    # fig, ax = plt.subplots()
    # # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.scatter(ReachedStrike['exportedAt'], ReachedStrike['priceDiffPerc'], s = 7, color='g', alpha=0.7, label='Did reach strike')
    # ax.scatter(notReachedStrike['exportedAt'], notReachedStrike['priceDiffPerc'], s = 7, color='r', alpha=0.7,
    #            label='Not reached strike')
    # ax.legend(loc="upper right")
    # ax.set_xlabel('Export date')
    # ax.xticks(rotation=90)
    # ax.set_ylabel('Strike Price increase')
    # ax.set_title('All Call options plotted')
    # plt.show()
    # fig.savefig("scheduled_jobs/summary_content/scatter.png")


    #####################
    # Measure performance
    # AUC
    auc_roc = plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all test observations - {}'.format(model.version),type='roc')
    auc_pr = plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all test observations - {}'.format(model.version),type='pr')
    # Brier score
    brier_score = brier_score_loss(pred_df['actual'],pred_df['prob'])

    if ext_plots:
        # Confucion matrix
        showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

        # Calibration plot
        plotCalibrationCurve(pred_df['actual'], pred_df['prob'], title='all test observations - {}'.format(model.version), bins=10)

        # Show performance for different segments
        brackets = [{'lower':1.05, 'higher':1.1}
            , {'lower':1.1, 'higher':1.2}
            , {'lower':1.2, 'higher':9999}
                    ]

        for bracket in brackets:
            print('Strike price / stock price ratio between {} and {}'.format(bracket['lower'], bracket['higher']))

            title = '{} with ratio between {} and {}'.format(model.version, bracket['lower'], bracket['higher'])
            select_df = pred_df[(pred_df['priceDiffPerc'] >= bracket['lower']) & (pred_df['priceDiffPerc'] < bracket['higher'])].copy()
            print('Nr observations {} out of {} ({})'.format(len(select_df),len(pred_df),round(len(select_df)/len(pred_df),2)))

            auc_roc = plotCurveAUC(select_df['prob'],select_df['actual'], title = title,type='roc')
            auc_pr = plotCurveAUC(select_df['prob'],select_df['actual'], title = title,type='pr')
            print('AUC ROC: {}'.format(round(auc_roc,3)))
            # print('AUC PR: {}'.format(round(auc_pr,3)))
            plotCalibrationCurve(select_df['actual'], select_df['prob'], title = title, bins=10)

            simpleTradingStrategy(select_df, actualCol='actual', filterset={'minThreshold': 0.1, 'maxThreshold': 0.99, 'minDaysToExp': 3, 'maxDaysToExp': 60, 'minStrikeIncrease': 1.05}, title = title)


    print('AUC ROC: {}'.format(auc_roc))
    print('AUC PR: {}'.format(auc_pr))
    print('Brier score: {}'.format(brier_score))
    ###########
    # profitablity
    # calculate trading profit
    print("High probability trading")
    roi, _, _, _ = simpleTradingStrategy(pred_df, actualCol='actual', filterset={'minThreshold': 0.75, 'maxThreshold': 0.99, 'minDaysToExp': 3, 'maxDaysToExp': 60, 'minStrikeIncrease': 1.05}, title = 'High prob ' + model.version)
    print("Roi: {}\n".format(roi))

    print("High profitability trading")
    roi, _, _, _ = simpleTradingStrategy(pred_df, actualCol='actual', filterset={'minThreshold': 0.6, 'maxThreshold': 0.99, 'minDaysToExp': 3, 'maxDaysToExp': 60, 'minStrikeIncrease': 1.2}, title = 'High prof ' + model.version)
    print("Roi: {}\n".format(roi))

    ###########
    # Feature importance
    # from sklearn.inspection import permutation_importance
    # permutation_imp = permutation_importance(model, pred_df[model.feature_names], pred_df['actual'])
    # feat_permutation_imp = pd.DataFrame({'feature': model.feature_names, 'importance': permutation_imp.importances_mean}).sort_values('importance', ascending=False).reset_index(drop=True)
    # feat_impurity_imp = pd.DataFrame({'feature': model.feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)
    #
    # print("Feature permutation importance: \n{}\n\n".format(feat_permutation_imp.head(10)))
    # print("Feature impurity importance: \n{}".format(feat_impurity_imp))

if __name__=='__main__':
    ###########
    # Load data and model
    # set configurations
    model_name = 'DEV_c_GB64_v4x3'

    # Load test datasets
    df_test = pd.read_csv("data/validation/test_df.csv")
    df_oot = pd.read_csv("data/validation/oot_df.csv")

    # Load model
    getwd = os.getcwd()
    with open(getwd+'/trained_models/'+model_name+'.sav', 'rb') as file:
        model = pickle.load(file)

    model = pipe._final_estimator
    model.version = model_name
    model.feature_names = features
    modelPerformanceReport(model, df_oot, ext_plots=True)