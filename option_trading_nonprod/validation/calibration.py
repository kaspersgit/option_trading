# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def plotCalibrationCurve(actuals, probs, title, bins=10, returnfig=False, savefig=False, saveFileName='test.png', show_plot=False):
    """
    Plot the calibration curve for a set of true and predicted values

    :param actuals: true target value
    :param probs: predicted probability of target
    :param bins: how many bins to divide data in for plotting
    :param savefig: boolean if plot should be saved
    :param saveFileName: str path to which to save the plot
    :return: calibration plot
    """
    plt.figure(figsize=(10, 10))

    # below would overwrite the figure in a plot
    # plt.figure(0, figsize=(10, 10))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(actuals, probs, n_bins=bins)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='Predicted probs.')

    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve) - {}'.format(title))
    if show_plot:
        plt.show()
    if savefig:
        plt.savefig(saveFileName)
    if returnfig:
        return plt

def plotCalibrationCurvePlotly(actuals, probs, title, bins=10, returnfig=False, savefig=False, saveFileName='test.png'):
    """
    Plot the calibration curve for a set of true and predicted values

    :param actuals: true target value
    :param probs: predicted probability of target
    :param bins: how many bins to divide data in for plotting
    :param savefig: boolean if plot should be saved
    :param saveFileName: str path to which to save the plot
    :return: calibration plot
    """
    # summaries actuals and predicted probs to (bins) number of points
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(actuals, probs, n_bins=bins)

    # Create scatter plot
    fig = px.scatter(x=mean_predicted_value, y=fraction_of_positives, title='Calibration plot (reliability curve)')

    # Make trace be line plus dots
    fig.data[0].update(mode='markers+lines')

    # set axis range to 0 - 1
    fig.update_layout(xaxis_range=[0,1], yaxis_range=[0,1], xaxis_title='Predicted probability', yaxis_title='Fraction of positives')

    # Add diagonal reference line
    fig.add_shape(type="line",
                  xref="paper", yref="paper",
                  x0=0, x1=1, y0=0, y1=1,
                  line=dict(
                      color="black",
                      width=2,
                      dash="dot",
                  )
    )

    if savefig:
        fig.write_image(saveFileName)
        print(f'Created and saved calibration plot as {saveFileName}')

    if returnfig:
        return fig

def plot_calibration_curve_mult(est, X_train, y_train, X_test, y_test, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=5, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=5, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()