# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults
import os
from datetime import datetime, timedelta
import numpy as np

# suppress copy warning
pd.options.mode.chained_assignment = None  # default='warn'

# load in functions 
def level_enriching(df):
    df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
    df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
    df['inTheMoney'] = np.where((df['symbolType']=='Call') & (df['baseLastPrice'] >= df['strikePrice']),1,0)
    df['inTheMoney'] = np.where((df['symbolType']=='Putt') & (df['baseLastPrice'] <= df['strikePrice']),1,df['inTheMoney'])
    df['nrOptions'] = 1
    df['strikePriceCum'] = df['strikePrice']

    df.sort_values(['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ], inplace=True)

    df_stock = df[['exportedAt','baseSymbol','baseLastPrice']].drop_duplicates()

    df_symbol = df[['exportedAt','baseSymbol','symbolType','daysToExpiration','strikePrice','inTheMoney','volume','openInterest','volatility','lastPrice'
            ]].groupby(['exportedAt','baseSymbol','symbolType','inTheMoney'
            ]).agg({'baseSymbol':'count', 'strikePrice':'mean', 'volume':'sum', 'openInterest':'sum', 'daysToExpiration':'mean', 'lastPrice':'mean', 'volatility':'mean'
            }).rename(columns={'baseSymbol':'nrOptions', 'strikePrice':'meanStrikePrice', 'volume':'sumVolume','openInterest':'sumOpenInterest','daysToExpiration':'meanDaysToExpiration','lastPrice':'meanLastPrice', 'volatility':'meanVolatility'
            }).reset_index()
    
    itm = df['inTheMoney'].unique()
    symbol = df['symbolType'].unique()

    for s in symbol:
        for i in itm:
            if i == 1:
                itm_str = 'Itm'
            elif i == 0:
                itm_str = 'Otm'
            
            base_colname = s + itm_str
            temp_df = df_symbol[(df_symbol['symbolType'] == s) & (df_symbol['inTheMoney'] == i)]

            temp_df['volumeOIratio'] = temp_df['sumVolume'] / temp_df['sumOpenInterest']
            temp_df = temp_df.drop(['inTheMoney'], axis=1)
            temp_df.rename(columns={'nrOccurences':'nr' + base_colname, 'meanStrikePrice':'meanStrike' + base_colname
                , 'sumVolume':'volume' + base_colname, 'sumOpenInterest':'openInterest' + base_colname
                , 'meanDaysToExpiration':'daysToExpiration' + base_colname, 'volumeOIratio':'volumeOIratio' + base_colname
                , 'meanLastPrice':'lastprice' + base_colname, 'meanVolatility':'volatility' + base_colname
                , 'nrOptions':'count' + base_colname}, inplace=True)
            temp_df.drop(columns=['symbolType'], inplace=True)

            df_stock = pd.merge(df_stock, temp_df, how='left', on=['exportedAt','baseSymbol'])

    # set nr of occurences to 0 when NaN
    #df[['nrCalls','nrPuts','volumeCall','volumePut']] = df[['nrCalls','nrPuts','volumeCall','volumePut']].fillna(0)
    return(df_stock)

# %%
# Load newest data
today = datetime.today().strftime("%Y-%m-%d")
current_path = os.getcwd()
df = pd.read_csv(current_path+'/barchart_unusual_activity_'+today+'.csv')

# Add extra columns
df_stock = level_enriching(df)

# due to scraping taking time baseLastPrice changes a bit for the same stock
# causing multiple rows for the same stock
df_stock = df_stock.drop_duplicates(subset=['baseSymbol'], keep='last')
df_stock['predDate'] = today
df_stock['const'] = 1.0

#%%
# Load model and predict
model = LogitResults.load(current_path + '/modelLogitStock')
model_version = 'stockLogit_20200808'
# Select columns which are model needs as input but leave out the constant
cols = model.params.index

pred = model.predict(df_stock[cols])
df_stock['prediction'] = pred
df_stock['modelVersion'] = model_version
# %%
# Subsetting the predictions
threshold = 0.5
maxBasePrice = 200 
expectedIncrease = 1.3
daysAhead = 14

buy_advise = df_stock[(df_stock['prediction'] > threshold) & 
    (df_stock['baseLastPrice'] < maxBasePrice)]
buy_advise['expectedPrice'] = expectedIncrease * buy_advise['baseLastPrice']
buy_advise['expectedDate'] = (pd.to_datetime(buy_advise['exportedAt']) + timedelta(days=daysAhead)).dt.strftime('%Y-%m-%d')
buy_advise = buy_advise[['baseSymbol', 'baseLastPrice', 'expectedDate','expectedPrice', 'prediction','modelVersion']]
buy_advise = buy_advise.sort_values('prediction').reset_index(drop=True)

# %%
# Sending an email with the predictions
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configurations and content
recipients = ['kasperde@hotmail.com']
emaillist = [elem.strip().split(',') for elem in recipients]
msg = MIMEMultipart()
msg['Subject'] = "Stock level buy advise"
msg['From'] = 'k.sends.python@gmail.com'

html = """\
<html>
  <head></head>
  <body>
    {0}
    <hr>
    <h3> Configurations </h3>
    <p>
    Minimal threshold: {1} <br>
    Maximum stock price: {2} <br>
    Expected price increase: {3} <br>
    Expected time period: {4} days <br>
    </p
  </body>
</html>
""".format(buy_advise.to_html(),threshold,maxBasePrice,
    expectedIncrease,daysAhead)

part1 = MIMEText(html, 'html')
msg.attach(part1)

# Sending the email
import smtplib, ssl

port = 465  # For SSL
password = open("/home/pi/Documents/trusted/ps_gmail_send.txt", "r").read()

# Create a secure SSL context
context = ssl.create_default_context()

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(msg['From'], password)
    server.sendmail(msg['From'], emaillist , msg.as_string())


# %%
