import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dfFilterOnGivenSet(df, filterset={}):
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
	return(df_filtered)

def simpleTradingStrategy(df,filterset={}, plot=True):
	if 'stocksBought' not in df.columns:
		df['stocksBought'] = 100 / df['baseLastPrice']
	if 'cost' not in df.columns:
		df['cost'] = df['stocksBought'] * df['baseLastPrice']
	if 'revenue' not in df.columns:
		df['revenue'] = df['stocksBought'] * np.where(df['reachedStrikePrice'] == 1, df['strikePrice'], df['finalPrice'])
	if 'profit' not in df.columns:
		df['profit'] = df['revenue'] - df['cost']
	df_filtered = dfFilterOnGivenSet(df, filterset)
	df_profit = df_filtered[['prob','cost','revenue','profit']].groupby('prob').sum().reset_index().sort_values('prob', ascending=False).copy()
	df_profit['cumCost'] = df_profit['cost'].cumsum()
	df_profit['cumRevenue'] = df_profit['revenue'].cumsum()
	df_profit['cumProfit'] = df_profit['profit'].cumsum()
	df_profit['cumProfitPerc'] = df_profit['cumProfit'] / df_profit['cumCost']

	# If plot is True then plot the prob on x axis and cumulative return on investment on y axis
	if plot:
		fig = plt.figure()
		ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
		ax.plot(df_profit['prob'], df_profit['cumProfitPerc'])
		plt.title('Inverse cumulative profit per threshold')
		plt.xlabel('Predicted probability')
		plt.ylabel('Profit percentage')
		plt.show()

	# Return the return on investment
	cost = df_profit['cost'].sum()
	revenue = df_profit['revenue'].sum()
	profit = df_profit['profit'].sum()
	roi = (revenue - cost) / cost
	return roi, cost, revenue, profit
