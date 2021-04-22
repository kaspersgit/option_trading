import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
from requests.exceptions import ConnectionError

import platform
if platform.system() == 'Darwin':
	import yahooquery as yq


def getContractPrices(df, startDateCol = 'exportedAt', endDateCol = 'expirationDate', type='minmax', use_package='yf'):
	"""
	For each unique ticker (column name 'baseSymbol') it will extract the
	daily stock prices between export date and expiration date. Of this time series
	the Min price, Max price, first price (on export date) and last price (on expiration date)
	will be added as columns to the dataframe

	:param df: Must include 'baseSymbol' and the columns specified in startDateCol and endDateCol
	:param startDateCol
	:param endDateCol
	:param type
	:param use_package: which package to use for extracting stock prices either yfinance (yf) or yahooquery (yq)

	:return: df: Added several columns indicating the first, last, min and max price reached until strike date
	also the dates of all these events is added
	"""
	# Make copy to not adjust original
	df_ = df.copy()

	# converting somewhat date columns to date strings
	df_[startDateCol]=pd.to_datetime(df_[startDateCol]).dt.strftime('%Y-%m-%d')
	df_[endDateCol]=pd.to_datetime(df_[endDateCol]).dt.strftime('%Y-%m-%d')

	# exclude entries where export and expiration date are the same
	df_ = df_[df_[startDateCol] != df_[endDateCol]]

	# Final df to be filled
	contracts_enr = pd.DataFrame(
		columns=['baseSymbol', startDateCol, endDateCol])

	config_df = pd.DataFrame(columns=['baseSymbol', 'minDate', 'maxDate'])

	# vectorize dtype='U10' is enough to fit date in string format
	tickers = df_['baseSymbol'].unique()
	minDates = np.zeros(len(tickers), dtype='U10')
	maxDates = np.zeros(len(tickers), dtype='U10')

	for s in range(len(tickers)):
		temp_df = df_[df_['baseSymbol'] == tickers[s]]
		minDates[s] = temp_df[startDateCol].min()
		maxDates[s] = temp_df[endDateCol].max()
	config_df['baseSymbol'] = tickers
	config_df['minDate_org'] = minDates
	config_df['maxDate_org'] = maxDates

	# add slack day to both start and end date
	maxDates_org = pd.to_datetime(maxDates)
	maxDates_adj = maxDates_org + timedelta(days=1)
	maxDates_adj = maxDates_adj.strftime("%Y-%m-%d")
	minDates_org = pd.to_datetime(minDates)
	minDates_adj = minDates_org + timedelta(days=1)
	minDates_adj = minDates_adj.strftime("%Y-%m-%d")

	config_df['maxDates_adj'] = maxDates_adj
	config_df['minDates_adj'] = minDates_adj

	# Print status
	print('Unique tickers: {}'.format(len(tickers)))

	# For each symbol extract the stock price series
	baseSymbols = config_df['baseSymbol']
	minDates_org = config_df['minDate_org']
	maxDates_org = config_df['maxDate_org']


	# for index, row in config_df.iterrows():
	for t in range(len(baseSymbols)):
		if t % 100 == 0:
			print('Rows done: {}'.format(t))

		# Get historic prices using yfinance or yahooquery
		stock_df = extractHistoricPrices(baseSymbols[t], minDate=minDates_org[t], maxDate=maxDates_adj[t], use_package=use_package)

		# Check for empty dataframe
		if len(stock_df) == 0:
			continue
		elif len(stock_df) == 1:
			if 'delisted' in list(stock_df.values)[0]:
				continue

		# Always include day of expiration
		stock_df = stock_df[stock_df.index >= minDates_org[t]]
		stock_df = stock_df[stock_df.index <= maxDates_org[t]]

		contracts = df_[df_['baseSymbol'] == baseSymbols[t]][['baseSymbol', startDateCol, endDateCol]]
		contracts.drop_duplicates(inplace=True)

		if type == 'minmax':
			df_minmax = pd.DataFrame(columns=['baseSymbol', 'exportedAt', 'expirationDate', 'minPrice', 'maxPrice',
											  'finalPrice', 'firstPrice', 'minPriceDate', 'maxPriceDate',
											  'finalPriceDate', 'firstPriceDate'])
			# For every different option contract get the prices
			for _index, contract_row in contracts.iterrows():
				wanted_timeserie = stock_df[contract_row[startDateCol]:contract_row[endDateCol]]
				# Check if time series is incomplete
				if wanted_timeserie.empty:
					print('Timeseries for {} is empty'.format(contract_row.baseSymbol))
					continue

				minPrice, maxPrice, finalPrice, firstPrice, minPriceDate, maxPriceDate, finalPriceDate, firstPriceDate = getMinMaxLastFirst(
					wanted_timeserie)
				info_dict = {'baseSymbol': contract_row['baseSymbol'], startDateCol: contract_row[startDateCol], endDateCol:contract_row[endDateCol],
							 'minPrice': minPrice, 'maxPrice': maxPrice, 'finalPrice': finalPrice, 'firstPrice': firstPrice,
							 'minPriceDate': minPriceDate, 'maxPriceDate': maxPriceDate, 'finalPriceDate': finalPriceDate, 'firstPriceDate': firstPriceDate}

				df_minmax = df_minmax.append(info_dict, ignore_index=True)

			# Merge extracted prices with contracts table
			contracts = contracts.merge(df_minmax, how='left',
									left_on=['baseSymbol', startDateCol, endDateCol], right_on=['baseSymbol', startDateCol, endDateCol])

		if type == 'indicators':
			# Add technical indicators
			df_indicators = getTechnicalIndicators(stock_df, indicators=['macd','rsi','obv','bbands'], fast=2, slow=20)
			df_indicators.index = df_indicators.index + timedelta(days=1)
			df_indicators.index = df_indicators.index.astype(str)
			# Merge extracted indicators with contracts table
			contracts = contracts.merge(df_indicators.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj_close', 'Volume'], errors='ignore'), how='left',
										left_on=endDateCol, right_index=True)

		# Add contract to master df where stock time series is found
		if not contracts.empty:
			# contracts = contracts[contracts['maxPrice'].notna()]
			contracts_enr = contracts_enr.append(contracts, ignore_index=True)

	return contracts_enr

def extractHistoricPrices(ticker, minDate, maxDate, use_package='yf'):
	print(f"Trying stock: {ticker}")
	if use_package == 'yf':
		stock_df = yf.download(ticker, start=minDate, end=maxDate)
	if use_package == 'yq':
		try:
			stock_df = yq.Ticker(ticker).history(start=minDate, end=maxDate)
		except ConnectionError as err:
			print('Connection error, waiting 5 seconds')
			time.sleep(5)
	return stock_df

def getTechnicalIndicators(stock_df, indicators=['rsi', 'obv', 'bbands'], fast=5, slow=20):
	"""
	For a set time period give the technical indicators on the final time period
	An overview of all available indicators: run pd.DataFrame().ta.indicators()

	:param stock_df: pandas dataframe with only Date as index
	:return: Any of the indicators specified in a pandas dataframe
	"""
	# Create copy to not adjust original
	stock_df_ = stock_df.copy()

	if len(stock_df_) > slow - 10:
		# Add different indicators
		for i in indicators:
			class_method = getattr(stock_df.ta, i)

			try:
				result = class_method(fast=fast, slow=slow)
			except:
				print("Data not sufficient to calculate {}".format(i))
				continue
			else:
				stock_df_ = stock_df_.join(result)
	else:
		print("Stock data going {} days back not available".format(slow+10))

	return (stock_df_)


def getMinMaxLastFirst(stock_df):
	"""
	For the period from scraping the option data until execution date
	 get the minimum, maximum, last and first price of the stock

	:param stock_df: pandas dataframe with only Date as index
	:return: Lowest, highest, last and first price in period together with corresponding dates
	"""
	# make sure df is ordered on time
	minPrice = stock_df['Low'][1::].min()
	maxPrice = stock_df['High'][1::].max()
	finalPrice = stock_df['Close'].iloc[-1]
	firstPrice = stock_df['Close'].iloc[0]

	minPriceDate = stock_df[stock_df['Low'] == minPrice].index.min()
	maxPriceDate = stock_df[stock_df['High'] == maxPrice].index.min()
	finalPriceDate = stock_df.index[-1]
	firstPriceDate = stock_df.index[0]

	return (minPrice, maxPrice, finalPrice, firstPrice, minPriceDate, maxPriceDate, finalPriceDate, firstPriceDate)


def limitDaysToExpiration(df, min=15, max=25):
	df = df[(df['daysToExpiration'] > min) & (df['daysToExpiration'] < max)]
	return (df)

def getCurrentStockPrice(ticker, attribute='Close'):
	print(ticker)

	if isinstance(ticker, np.ndarray):
		ticker = ticker.tolist()
		print("Converted numpy ndarray to list")

	data = yf.download(ticker, period='15m', interval='5m')
	if len(data) == 0:
		result = 9999
	else:
		returned_value = data[attribute].values
		result = returned_value[-1]
	return result

def batch_enrich_df(df, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney']):
	"""
	Adding information about the other option data within the same batch
	together with some ratios
	:param df: must include ['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'strikePrice', 'openInterest', 'volume']
	:return:
	"""
	merge_cols = groupByColumns.copy()
	merge_cols.remove('symbolType')

	df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
	df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
	df['inTheMoney'] = np.where((df['symbolType'] == 'Call') & (df['baseLastPrice'] >= df['strikePrice']), 1, 0)
	df['inTheMoney'] = np.where((df['symbolType'] == 'Putt') & (df['baseLastPrice'] <= df['strikePrice']), 1,
								df['inTheMoney'])
	df['nrOptions'] = 1
	df['strikePriceCum'] = df['strikePrice']

	df['volumeTimesStrike'] = df['strikePrice'] * df['volume']

	df.sort_values(['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'strikePrice'
					], inplace=True)

	df_symbol = df[['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'strikePrice', 'inTheMoney', 'volume',
					'openInterest', 'volumeTimesStrike'
					]].groupby(groupByColumns).agg(
		{'baseSymbol': 'count', 'strikePrice': 'mean', 'volume': 'sum', 'openInterest': 'sum', 'volumeTimesStrike': 'sum'
		 }).rename(columns={'baseSymbol': 'nrOccurences', 'strikePrice': 'meanStrikePrice', 'volume': 'sumVolume', 'openInterest': 'sumOpenInterest', 'volumeTimesStrike': 'sumVolumeTimesStrike'
							}).reset_index()
	df_symbol['weightedStrike'] = df_symbol['sumVolumeTimesStrike'] / df_symbol['sumVolume']

	# only give info about calls with higher strike price
	df_option_inv_cum = df[
		['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'strikePrice', 'strikePriceCum', 'inTheMoney',
		 'volume', 'openInterest', 'nrOptions'
		 ]].groupby(groupByColumns + ['strikePrice']).sum().sort_values('strikePrice', ascending=False
										  ).groupby(
		groupByColumns
	).agg({'volume': 'cumsum', 'openInterest': 'cumsum', 'nrOptions': 'cumsum', 'strikePriceCum': 'cumsum'
		   }).rename(
		columns={'volume': 'volumeCumSum', 'openInterest': 'openInterestCumSum', 'nrOptions': 'nrHigherOptions',
				 'strikePriceCum': 'higherStrikePriceCum'
				 }).reset_index()

	# Excluding own option from the higher options
	df_option_inv_cum['nrHigherOptions'] = df_option_inv_cum['nrHigherOptions'] - 1
	df_option_inv_cum['higherStrikePriceCum'] = df_option_inv_cum['higherStrikePriceCum'] - df_option_inv_cum[
		'strikePrice']

	df_call = df_symbol[df_symbol['symbolType'] == 'Call'].copy()
	df_call.rename(columns={'nrOccurences': 'nrCalls', 'meanStrikePrice': 'meanStrikeCall', 'sumVolume': 'sumVolumeCall',
							'sumOpenInterest': 'sumOpenInterestCall', 'sumVolumeTimesStrike': 'sumVolumeTimesStrikeCall',
							'weightedStrike': 'weightedStrikeCall'}, inplace=True)
	df_call.drop(columns=['symbolType'], inplace=True)

	df_put = df_symbol[df_symbol['symbolType'] == 'Put'].copy()
	df_put.rename(columns={'nrOccurences': 'nrPuts', 'meanStrikePrice': 'meanStrikePut', 'sumVolume': 'sumVolumePut',
						   'sumOpenInterest': 'sumOpenInterestPut', 'sumVolumeTimesStrike': 'sumVolumeTimesStrikePut',
						   'weightedStrike': 'weightedStrikePut'}, inplace=True)
	df_put.drop(columns=['symbolType'], inplace=True)

	# Add summarized data from Calls and Puts to df
	df = pd.merge(df, df_call, how='left', on=merge_cols)
	df = pd.merge(df, df_put, how='left', on=merge_cols)
	df = pd.merge(df, df_option_inv_cum, how='left',
				  on=groupByColumns + ['strikePrice'])

	df['meanStrikeCallPerc'] = df['meanStrikeCall'] / df['baseLastPrice']
	df['meanStrikePutPerc'] = df['meanStrikePut'] / df['baseLastPrice']
	df['midpointPerc'] = df['midpoint'] / df['baseLastPrice']
	df['meanHigherStrike'] = np.where((df['higherStrikePriceCum'] > 0) & (df['nrHigherOptions'] > 0),
									  df['higherStrikePriceCum'] / df['nrHigherOptions'], 0)

	# Set to 0 when NaN
	df.fillna({'nrCalls': 0, 'nrPuts': 0,
			   'sumVolumeCall': 0, 'sumVolumePut': 0,
			   'sumOpenInterestCall': 0, 'sumOpenInterestPut': 0,
			   'sumVolumeTimesStrikeCall': 0, 'sumVolumeTimesStrikePut':0,
			   'weightedStrikeCall': 0, 'weightedStrikePut': 0}, inplace=True)
	cols = ['meanStrikeCall', 'meanStrikePut']
	for col in cols:
		df[col] = df.apply(
			lambda row: row['baseLastPrice'] if np.isnan(row[col]) else row[col],
			axis=1
		)
	cols = ['meanStrikeCallPerc', 'meanStrikePutPerc']
	for col in cols:
		df[col] = df.apply(
			lambda row: 0 if np.isnan(row[col]) else row[col],
			axis=1
		)

	return (df)

if __name__ == '__main__':
	print('Empty test run')
	# df_prices = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')