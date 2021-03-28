import optuna
from optuna.visualization import plot_optimization_history, plot_slice
from optuna.visualization import plot_param_importances

# to actually show plot
import plotly.io as pio
pio.renderers.default = "browser"

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, make_scorer

# Have to manually pre create the X_train, y_train, X_val and y_val before running
# and make sure the correct features are selected
X_val.fillna(X_val.mean(), inplace=True)
X_train.fillna(X_train.mean(), inplace=True)

def objective(trial):

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
		"n_estimators": trial.suggest_int("n_estimators", 500,2000),
		"max_depth": trial.suggest_int("max_depth", 4, 8),
		"max_features": trial.suggest_int("max_features", 10, 14),
		"min_samples_split": trial.suggest_int("min_samples_split", 200, 300),
		"subsample": trial.suggest_float("subsample", 0.7, 0.9),
		"learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
		"random_state": 42,
	}

	# Specify model, fit, predict and produce score
	model = GradientBoostingClassifier()
	model.set_params(**params)
	model.fit(X_train[top30features], y_train)
	prob = model.predict_proba(X_val[top30features])[:,1]
	roc_auc = roc_auc_score(y_val, prob)
	return 1 - roc_auc

if __name__ == '__main__':

	study = optuna.create_study()
	study.optimize(objective, n_trials=50)
	fig = optuna.visualization.plot_optimization_history(study)
	fig.show()
	fig = optuna.visualization.plot_param_importances(study)
	fig.show()

	print(study.best_params)