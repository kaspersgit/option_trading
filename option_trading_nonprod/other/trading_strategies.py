import pandas as pd
import matplotlib.pyplot as plt

def simpleTradingStrategy(df,filterset={}, plot=True):
	# Fill up filterset with default in case of non existing
	if "threshold" not in filterset:
		filterset['threshold'] = 0.0
	if "maxBasePrice" not in filterset:
		filterset['maxBasePrice'] = 200
	if "minDaysToExp" not in filterset:
		filterset['minDaysToExp'] = 2
	if "maxDaysToExp" not in filterset:
		filterset['maxDaysToExp'] = 100
	if "minStrikeIncrease" not in filterset:
		filterset['minStrikeIncrease'] = 1.0
	print('Filtering data set on following rules: \n{}'.format(filterset))

	# in case column did not exist yet
	df['strikePricePerc'] = df['strikePrice'] / df['baseLastPrice']

	df_filtered = df[(df['prob'] > filterset['threshold']) &
		(df['symbolType'] == 'Call') &
		(df['daysToExpiration'] < filterset['maxDaysToExp']) &
		(df['strikePricePerc'] > filterset['minStrikeIncrease']) &
		(df['daysToExpiration'] > filterset['minDaysToExp']) &
		(df['baseLastPrice'] < filterset['maxBasePrice'])].copy()

	df_profit = df_filtered[['prob','cost','revenue','profit']].groupby('prob').sum().reset_index().sort_values('prob', ascending=False).copy()
	df_profit['cumCost'] = df_profit['cost'].cumsum()
	df_profit['cumRevenue'] = df_profit['revenue'].cumsum()
	df_profit['cumProfit'] = df_profit['profit'].cumsum()
	df_profit['cumProfitPerc'] = df_profit['cumProfit'] / df_profit['cumCost']

	# If plot is True then plot the prob on x axis and cumulative return on investment on y axix
	if plot:
		fig = plt.figure()
		ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
		ax.plot(df_profit['prob'], df_profit['cumProfitPerc'])
		plt.show()

	# Return the return on investment
	roi = (df_profit['revenue'].sum() - df_profit['cost'].sum()) / df_profit['cost'].sum()
	cost = df_profit['cost'].sum()
	revenue = df_profit['revenue'].sum()
	profit = df_profit['profit'].sum()
	return roi, cost, revenue, profit
