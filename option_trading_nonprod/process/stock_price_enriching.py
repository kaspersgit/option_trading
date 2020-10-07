import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

def get_mature_options(df, symbolType=['Call'], minDays = 4, maxDays = 18):
    # Select only mature cases (and exclude options with less then 5 days to expiration)
    df = df[pd.to_datetime(df['expirationDate']) < datetime.today()]
    df = df[(df['daysToExpiration'] > minDays) & (df['daysToExpiration'] < maxDays)]
    df = df[df['symbolType'].isin(symbolType)]
    return(df)

def get_interesting_options(df, minPrice = 0, maxPrice = 200):
    # In the money based on the last base price
    df=df[df['inTheMoney']!=1]
    # In the money based on the 1.025 * baseLastPrice 
    df=df[(~df['nextBDopen'].isnull()) & (df['strikePrice']> 1.025*df['baseLastPrice'])]
    # Stock price lower than 200 $
    df=df[df['baseLastPrice'] < maxPrice]
    # Return result
    return(df)

def add_weekday_dummies(df):
    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, x in enumerate(names):
        df[x] = (df.index.get_level_values(0).weekday == i).astype(int)
    return(df)

def add_stock_price(df):
    ## Adding actual stockprices
    # create empty list
    final_df = []
    # Loop through all different scraping date
    start_dates = df['exportedAt'].unique()
    for start_date in start_dates:
        data_df = df[df['exportedAt']==start_date]
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
            tickers_list = data_df[data_df['expirationDate']==end]['baseSymbol'].unique()
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
        
            if len(tickers_list)==1:
                highs = pd.DataFrame({'baseSymbol': [tickers], 'maxPrice': [highs]})
                lows = pd.DataFrame({'baseSymbol': [tickers], 'minPrice': [lows]})
                openbd = pd.DataFrame({'baseSymbol': [tickers], 'nextBDopen': [openbd]})
                last_close = pd.DataFrame({'baseSymbol': [tickers], 'lastClose': [last_close]})
            else:
                highs = highs.reset_index()
                highs.columns=['baseSymbol','maxPrice']
                lows = lows.reset_index()
                lows.columns=['baseSymbol','minPrice']
                openbd = openbd.reset_index()
                openbd.columns=['baseSymbol','nextBDopen']
                last_close = last_close.reset_index()
                last_close.columns=['baseSymbol','lastClose']

            #temp_df = pd.merge(temp_df, highs, how='left', on='baseSymbol')
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
    return(final_df)

def last_10d_avg(df):
    ## Adding actual stockprices
    # create empty list
    final_df = []
    # Loop through all different scraping date
    end_dates = df['exportedAt'].unique()
    for end_date in end_dates:
        data_df = df[df['exportedAt']==end_date]
        # Get the different dates
        end_date = pd.to_datetime(end_date)
        end_date_m10 = (end_date - pd.DateOffset(10)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        print('Working with data scraped on {}'.format(end_date))
            
        tickers_list = data_df[data_df['exportedAt']==end_date]['baseSymbol'].unique()
        tickers = ','.join(tickers_list)
        data = yf.download(tickers, start=end_date_m10, end=end_date)

        # Get max high and min low
        highs = data.loc[end_date_m10::]['High'].max()
        lows = data.loc[end_date_m10::]['Low'].min()
        means = data['Close'].mean()
    
        if len(tickers_list)==1:
            highs = pd.DataFrame({'baseSymbol': [tickers], 'maxPrice': [highs]})
            lows = pd.DataFrame({'baseSymbol': [tickers], 'minPrice': [lows]})
            means = pd.DataFrame({'baseSymbol': [tickers], 'meanLast10D': [means]})
        else:
            highs = highs.reset_index()
            highs.columns=['baseSymbol','maxPrice']
            lows = lows.reset_index()
            lows.columns=['baseSymbol','minPrice']
            means = means.reset_index()
            means.columns=['baseSymbol','meanLast10D']

        #temp_df = pd.merge(temp_df, highs, how='left', on='baseSymbol')
        temp_df = pd.merge(highs, lows, how='left', on=['baseSymbol'])
        temp_df = pd.merge(temp_df, means, how='left', on=['baseSymbol'])
        temp_df['exportedAt'] = end_date
        if len(final_df) == 0:
            final_df = temp_df
        else:
            final_df = final_df.append(temp_df)
    final_df.reset_index(drop=True, inplace=True)
    return(final_df)
