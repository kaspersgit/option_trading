import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filterDFonConstraints(df, config):
	# TODO fix this to make it perform the same as fuction below
	# Getting number of applicable accounts based on constraints
	# find constraints
	df_ = df.copy()

	# Fill up filterset with default in case of non existing
	if "minThreshold" not in config:
		config['minThreshold'] = 0.0
	if "maxThreshold" not in config:
		config['maxThreshold'] = 1.0
	if "maxBasePrice" not in config:
		config['maxBasePrice'] = 200
	if "minDaysToExp" not in config:
		config['minDaysToExp'] = 2
	if "maxDaysToExp" not in config:
		config['maxDaysToExp'] = 100
	if "minStrikeIncrease" not in config:
		config['minStrikeIncrease'] = 1.0
	# In case we have not scored yet
	if "minThreshold" not in config:
		config['minThreshold'] = 0.0
	if "prob" not in df_.keys():
		df_['prob'] = 1.0

	for col in df_.keys().str.lower():
		for con in [k.lower() for k in config.keys()]:
			if col in con:
				# Apply min value if existing
				if 'min' in con:
					df_ = df_[df_[col] >= config[con]]
				# Apply max value if existing
				elif 'max' in con:
					df_ = df_[df_[col] < config[con]]
				# Apply equality  if existing
				else:
					df_ = df_[df_[col] == config[con]]
	return df_

def dfFilterOnGivenSetOptions(df, filterset={}, type='stock'):
	df_ = df.copy()
	# Fill up filterset with default in case of non existing
	if "minThreshold" not in filterset:
		filterset['minThreshold'] = 0.0
	if "maxThreshold" not in filterset:
		filterset['maxThreshold'] = 1.0
	if "maxBasePrice" not in filterset:
		filterset['maxBasePrice'] = 200
	if "minDaysToExp" not in filterset:
		filterset['minDaysToExp'] = 2
	if "maxDaysToExp" not in filterset:
		filterset['maxDaysToExp'] = 100
	if "minStrikeIncrease" not in filterset:
		filterset['minStrikeIncrease'] = 1.0
	# In case we have not scored yet
	if "minThreshold" not in filterset:
		filterset['minThreshold'] = 0.0
	if "prob" not in df_:
		df_['prob'] = 1.0
	print('Filtering data set on following rules: \n{}'.format(filterset))
	# in case column did not exist yet
	df_['strikePricePerc'] = df_['strikePrice'] / df_['baseLastPrice']

	if type == 'stock':
		df_filtered = df_[(df_['prob'] > filterset['minThreshold']) &
			(df_['prob'] <= filterset['maxThreshold']) &
			(df_['symbolType'] == 'Call') &
			(df_['daysToExpiration'] < filterset['maxDaysToExp']) &
			(df_['strikePricePerc'] > filterset['minStrikeIncrease']) &
			(df_['daysToExpiration'] > filterset['minDaysToExp']) &
			(df_['baseLastPrice'] < filterset['maxBasePrice'])].copy()
	elif type == 'option':
		df_filtered = df_[(df_['prob'] > filterset['minThreshold']) &
						 (df_['prob'] <= filterset['maxThreshold']) &
						 (df_['symbolType'] == 'Call') &
						 (df_['daysToExpiration'] < filterset['maxDaysToExp']) &
						 (df_['strikePricePerc'] > filterset['minStrikeIncrease']) &
						 (df_['daysToExpiration'] > filterset['minDaysToExp']) &
						 (df_['lastPrice'] < filterset['maxBasePrice'])].copy()
	return(df_filtered)

def simpleTradingStrategy(df, actualCol = 'reachStrikePrice',filterset={}, plot=True, title='', savefig=False, saveFileName='test.png'):
	df_ = df.copy()
	if 'stocksBought' not in df_.columns:
		df_['stocksBought'] = 100 / df_['baseLastPrice']
	if 'cost' not in df_.columns:
		df_['cost'] = df_['stocksBought'] * df_['baseLastPrice']
	if 'revenue' not in df_.columns:
		df_['revenue'] = df_['stocksBought'] * np.where(df_[actualCol] == 1, df_['strikePrice'], df_['finalPrice'])
	if 'profit' not in df_.columns:
		df_['profit'] = df_['revenue'] - df_['cost']
	df_filtered = dfFilterOnGivenSetOptions(df_, filterset)
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
		plot_title = 'Profit per threshold ' + title
		plt.title(plot_title)
		plt.xlabel('Predicted probability')
		plt.ylabel('Profit percentage')
		plt.show()
		if savefig:
			plt.savefig(saveFileName)

	# Return the return on investment
	cost = df_profit['cost'].sum()
	revenue = df_profit['revenue'].sum()
	profit = df_profit['profit'].sum()
	roi = (revenue - cost) / cost
	return roi, cost, revenue, profit

def simpleTradingStrategyOptions(df, actualCol = 'reachedStrikePrice',filterset={'maxBasePrice': 3}, plot=True, title='- option strategy', savefig=False, saveFileName='profitabilityOptions.png'):
	df_ = df.copy()
	# TODO make sure all have an estimated option price
	df_ = df_[~df_['expOptionPrice'].isnull()]
	if 'optionsBought' not in df_.columns:
		df_['optionsBought'] = 100 / df_['lastPrice']
	if 'optionCost' not in df_.columns:
		df_['optionCost'] = df_['optionsBought'] * df_['lastPrice']
	if 'optionRevenue' not in df_.columns:
		df_['optionRevenue'] = df_['optionsBought'] * np.where(df_[actualCol] == 1, df_['expOptionPrice'], 0) # TODO putting zero here is harsh
	if 'optionProfit' not in df_.columns:
		df_['optionProfit'] = df_['optionRevenue'] - df_['optionCost']
	df_filtered = dfFilterOnGivenSet(df_, filterset, type = 'option')
	df_profit = df_filtered[['prob','optionCost','optionRevenue','optionProfit']].groupby('prob').sum().reset_index().sort_values('prob', ascending=False).copy()
	df_profit['cumCost'] = df_profit['optionCost'].cumsum()
	df_profit['cumRevenue'] = df_profit['optionRevenue'].cumsum()
	df_profit['cumProfit'] = df_profit['optionProfit'].cumsum()
	df_profit['cumProfitPerc'] = df_profit['cumProfit'] / df_profit['cumCost']

	# If plot is True then plot the prob on x axis and cumulative return on investment on y axis
	if plot:
		fig = plt.figure()
		ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
		ax.plot(df_profit['prob'], df_profit['cumProfitPerc'])
		plot_title = 'Profit per threshold ' + title
		plt.title(plot_title)
		plt.xlabel('Predicted probability')
		plt.ylabel('Profit percentage')
		plt.show()
		if savefig:
			plt.savefig(saveFileName)

	# Return the return on investment
	cost = df_profit['optionCost'].sum()
	revenue = df_profit['optionRevenue'].sum()
	profit = df_profit['optionProfit'].sum()
	roi = (revenue - cost) / cost
	return roi, cost, revenue, profit
