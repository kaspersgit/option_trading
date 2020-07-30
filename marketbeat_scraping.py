# %%
## Checking out https://www.barchart.com/options/unusual-activity/stocks?page=1
# and https://www.marketbeat.com/market-data/unusual-call-options-volume/
# and https://marketchameleon.com/Reports/UnusualOptionVolumeReport

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import re
import time
import random
import platform
import requests

def scrapeMarketBeat(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    data = []
    table = soup.find('table')  # , attrs={'id':'opt_unusual_volume'})
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        first2cols = [ele.find_all('div') for ele in cols]
        first2cols = [ele.text.strip() for ele in first2cols[0]][-2::]
        cols = [ele.text.strip() for ele in cols]
        cols = first2cols + cols[1:len(cols)]
        data.append([ele for ele in cols if ele])  # Get rid of empty values

    #colNames = table.find_all('th')
    #colNames = [ele.text.strip() for ele in colNames]
    # Hardcoded as 'ticker' is not a column name
    colNames = ['ticker', 'company', 'callOptionVolume', 'avgOptionVolume', 'increaseRelative2Avg', 'avgStockVolume',
                'indicators']

    df = pd.DataFrame(data=data, columns=colNames)

    # Get date of data
    dateSentence = soup.select('#cphPageTitle_pnlTwo')[0].text.strip()
    match = re.search(r'(\d+/\d+/\d+)', dateSentence)
    dataDate = datetime.strptime(match[0], "%m/%d/%Y").strftime('%Y-%m-%d')

    df['dataDate'] = dataDate
    return (df)

def text2float(textnum):
    scales = ["hundred", "thousand", "million", "billion", "trillion"]

    number = float(re.findall("([0-9]+[.]?[0-9]+)", textnum)[0])
    for idx, word in enumerate(scales):
        numwords[word] = (10 ** (idx * 3 or 2), 0)

    for word in textnum.split():
        if word in scales:
            scale, increment = numwords[word]
            number = number * scale + increment
    return(number)

### Let the scraping start
# Set some variables
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
today = datetime.today().strftime("%Y-%m-%d")
print('Script started at {}'.format(now))
url = 'https://www.marketbeat.com/market-data/unusual-call-options-volume/'

df = scrapeMarketBeat(url)

# Cleaning and adding columns
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df['exportedAt'] = now
df['dataDate'] = pd.to_datetime(df['dataDate'])
df['callOptionVolume'] = df["callOptionVolume"].str.replace(",", "").str.replace('*', '').astype(float)
df['avgOptionVolume'] = df["avgOptionVolume"].str.replace(",", "").str.replace('*', '').astype(float)
df['increaseRelative2Avg'] = df["increaseRelative2Avg"].str.replace("%", "").str.replace('*', '').astype(float)
df['avgStockVolume'] = df["avgStockVolume"].str.replace(",", "").str.replace('*', '')

df['avgStockVolume'] = df['avgStockVolume'].str.replace(",", "").str.replace('*', '')
df['avgStockVolume'] = df['avgStockVolume'].apply(text2float)

# Saving file as CSV
df.to_csv('/home/pi/Documents/python_scripts/option_trading/marketbeat_call_activity_' + dataDate + '.csv',
                index=False)

print('Script finished at {}'.format(now))
