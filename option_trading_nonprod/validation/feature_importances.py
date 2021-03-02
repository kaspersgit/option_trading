import matplotlib.pyplot as plt


def featureImportance1(model, features, make_plot=True, plot_top=20):
	feat_imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)
	if make_plot:
		feat_imp_plot = feat_imp.head(plot_top)
		plt.bar(x=feat_imp_plot['feature'], height=feat_imp_plot['importance'])
		plt.xticks(rotation=90)
		plt.xlabel('Feature')
		plt.ylabel('Importance')
		plt.title('Feature importance')
	return feat_imp