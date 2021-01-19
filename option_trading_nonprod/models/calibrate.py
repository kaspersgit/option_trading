from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle

def calibrate_model(model , X_fit, y_fit, method, save_model, path, name):
	# calibrated classifier
	cal_model = CalibratedClassifierCV(model, cv='prefit', method=method)
	cal_model.fit(X_fit, y_fit)
	if hasattr(model, 'feature_names'):
		cal_model.feature_names = model.feature_names
	if hasattr(model, 'feature_importances_'):
		cal_model.feature_importances_ = model.feature_importances_
	if hasattr(model, 'train_data_shape'):
		cal_model.train_data_shape = model.train_data_shape
	cal_model.calibration_data_shape = X_fit.shape

	if save_model:
		# Save calibrated model
		save_to = '{}{}.sav'.format(path, name)
		pickle.dump(cal_model, open(save_to, 'wb'))
		print('Saved model to {}'.format(save_to))

	return cal_model