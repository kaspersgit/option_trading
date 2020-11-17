# Load in all csv files from source folder
import os
import pandas as pd

# Temp setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set path
cwd = os.getcwd()
directory = os.path.join(cwd,"data/barchart")

# create empty df
df = pd.DataFrame()
# loop through all csv files and concatonate
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           f = pd.read_csv(os.path.join(root,file))
           #  Concatenate files into one pandas df
           df = pd.concat([df,f])
df.reset_index(drop=True, inplace=True)

# Select columns
df = df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice', 'expirationDate', 'daysToExpiration', 'bidPrice', 'midpoint', 'askPrice', 'lastPrice', 'volume', 'openInterest', 'volumeOpenInterestRatio', 'volatility', 'tradeTime', 'exportedAt']]


# Helper functions
def colType2Float(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    return clean_series

def colType2Int(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    clean_series = clean_series.astype('int')
    return clean_series

# Cast certain columns to float after cleaning
float_cols = ['baseLastPrice','strikePrice','bidPrice','midpoint','askPrice','lastPrice','volumeOpenInterestRatio','volatility']
for col in float_cols:
    df[col] = colType2Float(df[col])

# Changing date columns to datetime
date_cols = ['expirationDate','tradeTime','exportedAt']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Cast columns to int
int_cols = ['volume','openInterest']
for col in int_cols:
    df[col] = colType2Int(df[col])

# add target label
# stock price reached strike price at expiration date

# create funcions
def getMinMaxLastFirst(stock_df):
    """
    For the period from scraping the option data until execution date
     get the minimum, maximum, last and first price of the stock

    :param stock_df: pandas dataframe with only Date as index
    :return: Lowest, highest, last and first price in period
    """
    minPrice = stock_df['Low'].min()
    maxPrice = stock_df['High'].max()
    lastPrice = stock_df['Close'].iloc[-1]
    firstPrice = stock_df['Close'].iloc[0]
    return(minPrice, maxPrice, lastPrice, firstPrice)

def limitDaysToExpiration(df, min=15, max=25):
    df = df[(df['daysToExpiration'] > min) & (df['daysToExpiration'] < max)]
    return(df)

def getContractPrices(df):
    df = limitDaysToExpiration(df)
    contracts_enr = pd.DataFrame(columns=['baseSymbol','exportedAt','expirationDate','minPrice','maxPrice','finalPrice','firstPrice'])
    config_df = pd.DataFrame(columns=['baseSymbol','minDate','maxDate'])
    config_df['baseSymbol'] = df['baseSymbol'].unique()
    for symbol in config_df['baseSymbol']:
        temp_df = df[df['baseSymbol']==symbol]
        minDate = temp_df['exportedAt'].min()
        maxDate = temp_df['expirationDate'].max()
        config_df.at[config_df['baseSymbol']==symbol, 'minDate'] = minDate
        config_df.at[config_df['baseSymbol']==symbol, 'maxDate'] = maxDate

    # Print status
    print('Unique tickers: {}'.format(config_df['baseSymbol'].nunique()))
    import yfinance as yf

    # For each symbol extract the stock price series
    for index, row in config_df.iterrows():
        if index % 100 == 0:
            print('Rows done: {}'.format(index))
        stock_price = yf.download(row['baseSymbol'], start=row['minDate'], end=row['maxDate'])
        if len(stock_price) == 0:
            continue
        stock_price = stock_price[row['minDate']::]
        contracts = df[df['baseSymbol']==row['baseSymbol']][['baseSymbol','exportedAt','expirationDate']]
        contracts.drop_duplicates(inplace=True)
        # For every different option contract get the prices
        for index, contract_row in contracts.iterrows():
            minPrice, maxPrice, lastPrice, firstPrice = getMinMaxLastFirst(stock_price[contract_row['exportedAt']:contract_row['expirationDate']])
            contracts.at[(contracts['exportedAt']==contract_row['exportedAt']) & (contracts['expirationDate']==contract_row['expirationDate']),'minPrice'] = minPrice
            contracts.at[(contracts['exportedAt']==contract_row['exportedAt']) & (contracts['expirationDate']==contract_row['expirationDate']),'maxPrice'] = maxPrice
            contracts.at[(contracts['exportedAt']==contract_row['exportedAt']) & (contracts['expirationDate']==contract_row['expirationDate']),'lastPrice'] = lastPrice
            contracts.at[(contracts['exportedAt']==contract_row['exportedAt']) & (contracts['expirationDate']==contract_row['expirationDate']),'firstPrice'] = firstPrice
        # Add to master df
        contracts_enr = contracts_enr.append(contracts, ignore_index=True)

    return contracts_enr

# Using above functions
contracts_prices = getContractPrices(df)

# Filter df on only short time to expiration
df = limitDaysToExpiration(df)
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])

# Save enriched df as csv
df_enr.to_csv('data/barchart_yf_enr_1.csv')

