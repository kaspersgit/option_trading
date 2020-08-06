# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults
import os
from datetime import datetime
import numpy as np

# %%
# Load newest data
today = datetime.today().strftime("%Y-%m-%d")
current_path = os.getcwd()
df = pd.read_csv(current_path+'/barchart_unusual_activity_'+today+'.csv')


# Adding some additional columns
df['predDate'] = today
df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
df['inTheMoney'] = np.where(df['baseLastPrice'] >= df['strikePrice'],1,0)
df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','inTheMoney'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'
        ]).agg({'baseSymbol':'count', 'strikePrice':'mean'
        }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
        }).reset_index()
df = pd.merge(df,df_symbol, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'])

# cleaning columns
#cols = ['volatility']
#df[cols] = df[cols].apply(lambda x: x.str.replace(',',''))
#df[cols] = df[cols].apply(lambda x: x.str.replace('%',''))
#df[cols] = df[cols].apply(lambda x: x.astype('float'))

df['const'] = 1.0

#%%
# Load model and predict
model = LogitResults.load(current_path + '/modelLogit')
# Select columns which are model needs as input but leave out the constant
cols = model.params.index

pred = model.predict(df[cols])
df['prediction'] = pred

# %%
# Subsetting the predictions
threshold = 0.5
buy_advise = df[(df['prediction'] > threshold) & 
    (df['symbolType']=='Call') & 
    (df['daysToExpiration'] < 20) & 
    (df['priceDiffPerc'] > 1.05) & 
    (df['daysToExpiration'] > 3) & 
    (df['strikePrice'] < 200)]
buy_advise = buy_advise[['baseSymbol', 'predDate', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction']]
buy_advise = buy_advise.sort_values('priceDiffPerc').reset_index(drop=True)

# %%
# Sending an email with the predictions
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configurations and content
recipients = ['kasperde@hotmail.com','derekdh@gmail.com']
emaillist = [elem.strip().split(',') for elem in recipients]
msg = MIMEMultipart()
msg['Subject'] = "Stock buy advise"
msg['From'] = 'k.sends.python@gmail.com'

html = """\
<html>
  <head></head>
  <body>
    {0}
  </body>
</html>
""".format(buy_advise.to_html())

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
