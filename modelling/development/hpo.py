import optuna
from optuna.visualization import plot_optimization_history, plot_slice
from optuna.visualization import plot_param_importances

# to actually show plot
import plotly.io as pio
pio.renderers.default = "browser"

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, precision_recall_curve, auc

# Have to manually pre create the X_train, y_train, X_val and y_val before running
# and make sure the correct features are selected
X_val.fillna(X_val.mean(), inplace=True)
X_train.fillna(X_train.mean(), inplace=True)

def objective(trial, features, X_train, y_train, X_val, y_val):

	# specify range of parameters
	# params_init = {
	# 	"criterion": "friedman_mse",
	# 	"n_estimators": trial.suggest_int("n_estimators", 100, 100),
	# 	"max_depth": trial.suggest_int("max_depth", 4, 8),
	# 	"max_features": trial.suggest_int("max_features", 6, 14),
	# 	"min_samples_split": trial.suggest_int("min_samples_split", 200, 500),
	# 	"subsample": trial.suggest_float("subsample", 0.6, 1.0),
	# 	"learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-2, log=True),
	# 	"random_state": trial.suggest_int("random_state", 42, 42),
	# }

	params = {
		"criterion": "friedman_mse",
		"n_estimators": 3000,
		"max_depth": trial.suggest_int("max_depth", 2, 12),
		"max_features": trial.suggest_int("max_features", 2, 12),
		"min_samples_split": trial.suggest_int("min_samples_split", 50, 500),
		"subsample": trial.suggest_float("subsample", 0.7, 0.9),
		"learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
		"random_state": 42,
	}

	# Specify model, fit, predict and produce score
	# features = features_info
	model = GradientBoostingClassifier()
	model.set_params(**params)
	model.fit(X_train[features], y_train)
	prob = model.predict_proba(X_val[features])[:,1]
	pred = model.predict(X_val[features])
	roc_auc = roc_auc_score(y_val, prob)
	# precision recall auc
	yVar, xVar, thresholds = precision_recall_curve(y_val, prob)
	pr_auc = auc(xVar, yVar)
	precision = precision_score(y_val, pred)

	# as the package seems to want to minimize the score function we do 1 - the metric we want to maximize
	return 1 - precision

if __name__ == '__main__':

	study = optuna.create_study()
	study.optimize(lambda trial: objective(trial, features, X_train, y_train, X_val, y_val), n_trials=1)
	fig = optuna.visualization.plot_optimization_history(study)
	fig.show()
	fig = optuna.visualization.plot_param_importances(study)
	fig.show()

	print(study.best_params)