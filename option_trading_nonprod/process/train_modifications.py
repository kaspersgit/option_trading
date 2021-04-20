import pandas as pd

def getSampleWeights(df, column, normalize = True, squared = False):
	df_ = df.copy()
	df_[column] = pd.to_datetime(df_[column])
	df_.sort_values(column, ascending=True, inplace=True)
	ranking = df_.groupby('symbolType')[column].rank(ascending=True)
	if squared:
		ranking = ranking ** 2
	if normalize:
		weights = (ranking-ranking.min())/(ranking.max()-ranking.min())
	else:
		weights = ranking
	return weights