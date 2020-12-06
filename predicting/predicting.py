# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults
import os
from datetime import datetime
import numpy as np
import pickle

# %%
# Load newest data
today = datetime.today().strftime("%Y-%m-%d")
current_path = os.getcwd()
df = pd.read_csv(current_path+'/data/barchart/barchart_unusual_activity_'+today+'.csv')

# Select model
model_choice = 'AdaBoost'

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

df['const'] = 1.0

#%%
# Load model and predict
if model_choice == 'LogisticRegression':
    # Logistic Regression
    file_path = current_path + '/predicting/modelLogit'
    model = LogitResults.load(file_path)
    model_name = file_path.split('/')[-1]
    # Select columns which are model needs as input but leave out the constant
    features = model.params.index
    prob = model.predict(df[features])
elif model_choice == 'CatBoost':
    # Load CatBoost model
    import catboost as cb
    model = cb.CatBoostClassifier()
    file_path = current_path + '/trained_models/cb_v1.cbm'
    model.load_model(file_path, format='cbm')
    model_name = file_path.split('/')[-1]
    features = model.feature_names_
    prob = model.predict_proba(df[features])[:,1]
elif model_choice == 'AdaBoost':
	file_path = current_path + '/trained_models/c_AB_v1.sav'
	with open(file_path, 'rb') as file:
		model = pickle.load(file)
	model_name = file_path.split('/')[-1]
	features = model.feature_names
	prob = model.predict_proba(df[features])[:, 1]


df['prediction'] = prob
df['model'] = model_name

# %%
# Subsetting the predictions
threshold = 0.6
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 20
minStrikeIncrease = 1.05

buy_advise = df[(df['prediction'] > threshold) &
    (df['symbolType']=='Call') &
    (df['daysToExpiration'] < maxDaysToExp) &
    (df['priceDiffPerc'] > minStrikeIncrease) &
    (df['daysToExpiration'] > minDaysToExp) &
    (df['baseLastPrice'] < maxBasePrice)]
buy_advise = buy_advise[['baseSymbol', 'predDate', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction','model']]
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
    <hr>
    <h3> Configurations </h3>
    <p>
    Minimal threshold: {1} <br>
    Maximum stock price: {2} <br>
    Days to expiration between {3} and {4} <br>
    Strike price at least {5} higher than stock price <br>
    </p
  </body>
</html>
""".format(buy_advise.to_html(),threshold,maxBasePrice,
  minDaysToExp,maxDaysToExp,minStrikeIncrease)

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
