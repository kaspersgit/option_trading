import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import numpy as np


def get_mature_options(df, symbolType=['Call'], minDays=4, maxDays=18):
	# Select only mature cases (and exclude options with less then 5 days to expiration)
	df = df[pd.to_datetime(df['expirationDate']) < datetime.today()]
	df = df[(df['daysToExpiration'] > minDays) & (df['daysToExpiration'] < maxDays)]
	df = df[df['symbolType'].isin(symbolType)]
	return (df)


def get_interesting_options(df, minPrice=0, maxPrice=200):
	# In the money based on the last base price
	df = df[df['inTheMoney'] != 1]
	# In the money based on the 1.025 * baseLastPrice
	df = df[(~df['nextBDopen'].isnull()) & (df['strikePrice'] > 1.025 * df['baseLastPrice'])]
	# Stock price lower than 200 $
	df = df[df['baseLastPrice'] < maxPrice]
	# Return result
	return (df)


def add_weekday_dummies(df):
	names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
	for i, x in enumerate(names):
		df[x] = (df.index.get_level_values(0).weekday == i).astype(int)
	return (df)


def add_stock_price(df):
	## Adding actual stockprices
	# create empty list
	final_df = []
	# Loop through all different scraping date
	start_dates = df['exportedAt'].unique()
	for start_date in start_dates:
		data_df = df[df['exportedAt'] == start_date]
		# Get the different dates
		start_date = pd.to_datetime(start_date)
		start_date_p1 = (start_date + pd.DateOffset(1)).strftime('%Y-%m-%d')
		nextBD = (1 * pd.offsets.BDay() + start_date).strftime('%Y-%m-%d')
		start_date = start_date.strftime('%Y-%m-%d')
		print('Working with data scraped on {}'.format(start_date))

		# Get all different expiration dates
		expiration_dates = data_df['expirationDate'].unique()

		for end in expiration_dates:
			if end <= start_date_p1:
				continue
			print('Working with enddate {}'.format(end))
			tickers_list = data_df[data_df['expirationDate'] == end]['baseSymbol'].unique()
			tickers = ','.join(tickers_list)
			data = yf.download(tickers, start=start_date, end=end)

			# next business day opening
			# first check if avaialable
			if nextBD not in data.index:
				continue
			openbd = data.loc[nextBD]['Open']

			# Get max high and min low
			highs = data.loc[start_date_p1::]['High'].max()
			lows = data.loc[start_date_p1::]['Low'].min()
			last_close = data['Close'].tail(1).mean()

			if len(tickers_list) == 1:
				highs = pd.DataFrame({'baseSymbol': [tickers], 'maxPrice': [highs]})
				lows = pd.DataFrame({'baseSymbol': [tickers], 'minPrice': [lows]})
				openbd = pd.DataFrame({'baseSymbol': [tickers], 'nextBDopen': [openbd]})
				last_close = pd.DataFrame({'baseSymbol': [tickers], 'lastClose': [last_close]})
			else:
				highs = highs.reset_index()
				highs.columns = ['baseSymbol', 'maxPrice']
				lows = lows.reset_index()
				lows.columns = ['baseSymbol', 'minPrice']
				openbd = openbd.reset_index()
				openbd.columns = ['baseSymbol', 'nextBDopen']
				last_close = last_close.reset_index()
				last_close.columns = ['baseSymbol', 'lastClose']

			# temp_df = pd.merge(temp_df, highs, how='left', on='baseSymbol')
			temp_df = pd.merge(highs, lows, how='left', on=['baseSymbol'])
			temp_df = pd.merge(temp_df, openbd, how='left', on=['baseSymbol'])
			temp_df = pd.merge(temp_df, last_close, how='left', on=['baseSymbol'])
			temp_df['expirationDate'] = end
			temp_df['exportedAt'] = start_date
			if len(final_df) == 0:
				final_df = temp_df
			else:
				final_df = final_df.append(temp_df)
	final_df.reset_index(drop=True, inplace=True)
	return (final_df)


def last_10d_avg(df):
	## Adding actual stockprices
	# create empty list
	final_df = []
	# Loop through all different scraping date
	end_dates = df['exportedAt'].unique()
	for end_date in end_dates:
		data_df = df[df['exportedAt'] == end_date]
		# Get the different dates
		end_date = pd.to_datetime(end_date)
		end_date_m10 = (end_date - pd.DateOffset(10)).strftime('%Y-%m-%d')
		end_date = end_date.strftime('%Y-%m-%d')
		print('Working with data scraped on {}'.format(end_date))

		tickers_list = data_df[data_df['exportedAt'] == end_date]['baseSymbol'].unique()
		tickers = ','.join(tickers_list)
		data = yf.download(tickers, start=end_date_m10, end=end_date)

		# Get max high and min low
		highs = data.loc[end_date_m10::]['High'].max()
		lows = data.loc[end_date_m10::]['Low'].min()
		means = data['Close'].mean()

		if len(tickers_list) == 1:
			highs = pd.DataFrame({'baseSymbol': [tickers], 'maxPrice': [highs]})
			lows = pd.DataFrame({'baseSymbol': [tickers], 'minPrice': [lows]})
			means = pd.DataFrame({'baseSymbol': [tickers], 'meanLast10D': [means]})
		else:
			highs = highs.reset_index()
			highs.columns = ['baseSymbol', 'maxPrice']
			lows = lows.reset_index()
			lows.columns = ['baseSymbol', 'minPrice']
			means = means.reset_index()
			means.columns = ['baseSymbol', 'meanLast10D']

		# temp_df = pd.merge(temp_df, highs, how='left', on='baseSymbol')
		temp_df = pd.merge(highs, lows, how='left', on=['baseSymbol'])
		temp_df = pd.merge(temp_df, means, how='left', on=['baseSymbol'])
		temp_df['exportedAt'] = end_date
		if len(final_df) == 0:
			final_df = temp_df
		else:
			final_df = final_df.append(temp_df)
	final_df.reset_index(drop=True, inplace=True)
	return (final_df)


def getContractPrices(df, startDateCol = 'exportedAt', endDateCol = 'expirationDate', type='minmax'):
	"""
	For each unique ticker (column name 'baseSymbol') it will extract the
	daily stock prices between export date and expiration date. Of this time series
	the Min price, Max price, first price (on export date) and last price (on expiration date)
	will be added as columns to the dataframe

	:param df: Must include 'baseSymbol' and the columns specified in startDateCol and endDateCol
	:return: df: Added several columns indicating the first, last, min and max price reached until strike date
	also the dates of all these events is added
	"""
	# converting somewhat date columns to date strings
	df[startDateCol]=pd.to_datetime(df[startDateCol]).dt.strftime('%Y-%m-%d')
	df[endDateCol]=pd.to_datetime(df[endDateCol]).dt.strftime('%Y-%m-%d')

	# exclude entries where export and expiration date are the same
	df = df[df[startDateCol] != df[endDateCol]]

	contracts_enr = pd.DataFrame(
		columns=['baseSymbol', startDateCol, endDateCol])
	config_df = pd.DataFrame(columns=['baseSymbol', 'minDate', 'maxDate'])
	config_df['baseSymbol'] = df['baseSymbol'].unique()
	for symbol in config_df['baseSymbol']:
		temp_df = df[df['baseSymbol'] == symbol]
		minDate = temp_df[startDateCol].min()
		maxDate = temp_df[endDateCol].max()
		config_df.at[config_df['baseSymbol'] == symbol, 'minDate'] = minDate
		config_df.at[config_df['baseSymbol'] == symbol, 'maxDate'] = maxDate

	# Print status
	print('Unique tickers: {}'.format(config_df['baseSymbol'].nunique()))

	# For each symbol extract the stock price series
	for index, row in config_df.iterrows():
		if index % 100 == 0:
			print('Rows done: {}'.format(index))
		stock_price = yf.download(row['baseSymbol'], start=row['minDate'], end=row['maxDate'])
		# Check for empty dataframe
		if len(stock_price) == 0:
			continue
		stock_price = stock_price[row['minDate']::]
		contracts = df[df['baseSymbol'] == row['baseSymbol']][['baseSymbol', startDateCol, endDateCol]]
		contracts.drop_duplicates(inplace=True)

		if type == 'minmax':
			df_minmax = pd.DataFrame(columns=['baseSymbol', 'exportedAt', 'expirationDate', 'minPrice', 'maxPrice',
											  'finalPrice', 'firstPrice', 'minPriceDate', 'maxPriceDate',
											  'finalPriceDate', 'firstPriceDate'])
			# For every different option contract get the prices
			for _index, contract_row in contracts.iterrows():
				wanted_timeserie = stock_price[contract_row[startDateCol]:contract_row[endDateCol]]
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
				# fill in into df
				# for key in info_dict.keys():
				# 	contracts.at[(contracts[startDateCol] == contract_row[startDateCol]) & (
				# 			contracts[endDateCol] == contract_row[endDateCol]), key] = info_dict[key]

			# Merge extracted prices with contracts table
			contracts = contracts.merge(df_minmax, how='left',
									left_on=['baseSymbol', startDateCol, endDateCol], right_on=['baseSymbol', startDateCol, endDateCol])

		if type == 'indicators':
			# Add technical indicators
			df_indicators = getTechnicalIndicators(stock_price, indicators=['macd','rsi','obv','bbands'], fast=2, slow=4)
			df_indicators.index = df_indicators.index + timedelta(days=1)
			df_indicators.index = df_indicators.index.astype(str)
			# Merge extracted indicators with contracts table
			contracts = contracts.merge(df_indicators, how='left',
										left_on=endDateCol, right_index=True)

		# Add contract to master df where stock time series is found
		if not contracts.empty:
			# contracts = contracts[contracts['maxPrice'].notna()]
			contracts_enr = contracts_enr.append(contracts, ignore_index=True)

	return contracts_enr

def getTechnicalIndicators(stock_df, indicators=['rsi', 'obv', 'bbands'], fast=5, slow=20):
	"""
	For a set time period give the technical indicators on the final time period
	An overview of all available indicators: run pd.DataFrame().ta.indicators()

	:param stock_df: pandas dataframe with only Date as index
	:return: Any of the indicators specified
	"""
	if len(stock_df) > slow + 10:
		# Add different indicators
		for i in indicators:
			class_method = getattr(stock_df.ta, i)
			result = class_method(fast=fast, slow=slow)
			stock_df = stock_df.join(result)
	else:
		print("Stock data going {} days back not available \nFor stock: {}".format(slow+10,stock_df['baseSymbol']))

	return (stock_df)


def getMinMaxLastFirst(stock_df):
	"""
	For the period from scraping the option data until execution date
	 get the minimum, maximum, last and first price of the stock

	:param stock_df: pandas dataframe with only Date as index
	:return: Lowest, highest, last and first price in period together with corresponding dates
	"""
	# make sure df is ordered on time
	minPrice = stock_df['Low'].min()
	maxPrice = stock_df['High'].max()
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


def batch_enrich_df(df, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney']):
	"""
	Adding information about the other option data within the same batch
	together with some ratios
	:param df: must include ['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'strikePrice', 'openInterest', 'volume']
	:return:
	"""
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
	df = pd.merge(df, df_call, how='left', on=['exportedAt', 'baseSymbol', 'expirationDate', 'inTheMoney'])
	df = pd.merge(df, df_put, how='left', on=['exportedAt', 'baseSymbol', 'expirationDate', 'inTheMoney'])
	df = pd.merge(df, df_option_inv_cum, how='left',
				  on=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney', 'strikePrice'])

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