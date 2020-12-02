# Add meaningfull features

# import packages
from datetime import datetime, timedelta
import pandas as pd


# select target variable
# - add target label

# feature engineering
# - historic stock price related
# - historic options related
# - other company related information

# - historic stock price related
def stockPriceEvolutionLast2m(df):
	"""
	Extracts some basic statistics on the historic stock price of the last 2 months

	:param df: pandas dataframe with at least the columns baseSymbol and exportedAt
	:return: dataframe with added statistics
	"""

	# extract two needed columns
	df_clean = df[['baseSymbol','exportedAt']].drop_duplicates().reset_index(drop=True)
	df_clean['exportedAt_2m'] = df_clean['exportedAt'] - pd.DateOffset(months=2)
	df_clean['exportDate'] = pd.to_datetime(df_clean['exportedAt']).dt.strftime('%Y-%m-%d')
	df_clean['exportDate_2m'] = pd.to_datetime(df_clean['exportedAt_2m']).dt.strftime('%Y-%m-%d')

	# create master df
	df_last2m = pd.DataFrame(columns=['baseSymbol','exportedAt','expirationDate','minPrice','maxPrice','finalPrice','firstPrice'])
	# Create small df to reduce downloads
	config_df = pd.DataFrame(columns=['baseSymbol', 'minDate', 'maxDate'])
	config_df['baseSymbol'] = df_clean['baseSymbol'].unique()

	for symbol in config_df['baseSymbol']:
		temp_df = df_clean[df_clean['baseSymbol'] == symbol]
		minDate = temp_df['exportDate_2m'].min()
		maxDate = temp_df['exportDate'].max()
		config_df.at[config_df['baseSymbol'] == symbol, 'minDate'] = minDate
		config_df.at[config_df['baseSymbol'] == symbol, 'maxDate'] = maxDate

	# Get all unique export dates
	endDates = config_df['maxDate'].unique()
	startDates = config_df['minDate'].unique()

	# Loop through all export dates to do batch retrieval
	for start, end in zip(startDates, endDates):
		print('Working with start date {}\nand end date {}'.format(start, end))
		tickers_list = config_df[config_df['maxDate'] == end]['baseSymbol'].unique()
		tickers = ','.join(tickers_list)
		data = yf.download(tickers, start=start, end=end)

		# Add to master df
		contracts_enr = contracts_enr.append(contracts, ignore_index=True)
