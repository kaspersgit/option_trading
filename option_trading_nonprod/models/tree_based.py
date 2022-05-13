from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import pickle
import sys

if 32 << bool(sys.maxsize >> 32) == 64:
	# import xgboost as xgb
	import catboost as cb

# def fit_xgb(X_fit, y_fit, X_val, y_val, params, save_model, xgb_path, name):
# 	# Example
# 	# params={'max_depth':2,
# 	#                           'n_estimators':999999,
# 	#                           'colsample_bytree':0.3,
# 	#                           'learning_rate':0.02,
# 	#                           'objective': 'binary:logistic',
# 	#                           'n_jobs':-1}
# 	model = xgb.XGBClassifier()
# 	model.set_params(**params)
# 	model.fit(X_fit, y_fit,
# 			  eval_set=[(X_val, y_val)],
# 			  verbose=0,
# 			  early_stopping_rounds=1000)
#
# 	if save_model:
# 		# Save XGBoost Model
# 		save_to = '{}{}.model'.format(xgb_path, name)
# 		if hasattr(model, 'feature_names'): model.set_attr(feature_names='|'.join(model.feature_names))
# 		model.save_model(save_to)
# 		print('Saved model to {}'.format(save_to))
#
# 	return model
#
#
def fit_cb(X_fit, y_fit, X_val, y_val, params, save_model, cb_path, name):
	import catboost as cb
	# Example
	# params = {'iterations':100,
	#                               'max_depth':2,
	#                               'learning_rate':0.1,
	#                               'colsample_bylevel':0.03,
	#                               'objective':"Logloss"}
	model = cb.CatBoostClassifier()
	model.set_params(**params)
	model.fit(X_fit, y_fit,
			  eval_set=[(X_val, y_val)],
			  verbose=0,
              early_stopping_rounds=1000)
	model.feature_names = X_fit.columns
	model.train_data_shape = X_fit.shape
	model.train_data_describe = X_fit.describe()

	if save_model:
		# Save Catboost Model
		save_to = '{}{}.cbm'.format(cb_path, name)
		model.save_model(save_to)
		print('Saved model to {}'.format(save_to))

	return model

def fit_AdaBoost(X_fit, y_fit, X_val, y_val, params, save_model, ab_path, name):
	# Example
	# params = {'iterations':100,
	#                               'max_depth':2,
	#                               'learning_rate':0.1,
	#                               'colsample_bylevel':0.03,
	#                               'objective':"Logloss"}
	model = AdaBoostClassifier()
	model.set_params(**params)
	model.fit(X_fit, y_fit)
	model.feature_names = X_fit.columns
	model.train_data_shape = X_fit.shape
	model.train_data_describe = X_fit.describe()

	if save_model:
		# Save AdaBoost Model
		save_to = '{}{}.sav'.format(ab_path, name)
		pickle.dump(model, open(save_to, 'wb'))
		print('Saved model to {}'.format(save_to))

	return model

def fit_GBclf(X_fit, y_fit, X_val, y_val, params, save_model, gbc_path, name, **kwargs):
	# Example
	# params = {'iterations':100,
	#                               'max_depth':2,
	#                               'learning_rate':0.1,
	#                               'colsample_bylevel':0.03,
	#                               'objective':"Logloss"}
	model = GradientBoostingClassifier()
	model.set_params(**params)
	if 'sample_weight' in kwargs.keys():
		model.fit(X_fit, y_fit, sample_weight=kwargs['sample_weight'])
	else:
		model.fit(X_fit, y_fit)
	model.feature_names = X_fit.columns
	model.train_data_shape = X_fit.shape
	model.train_data_describe = X_fit.describe()

	if save_model:
		# Save GradientBoostingClassifier Model
		save_to = '{}{}.sav'.format(gbc_path, name)
		pickle.dump(model, open(save_to, 'wb'))
		print('Saved model to {}'.format(save_to))

	return model
