# Load packages
import pandas as pd

# Helper functions
def cherry_pick(df, OutOfMoney = 1.1, maxDTE = 10, minVolOIrate = 5, type = 'calls'):
    calls = (df['symbolType'] == 'Call') & (OutOfMoney * df['baseLastPrice'] < df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    putts = ( OutOfMoney * df['baseLastPrice'] > df['strikePrice']) & (df['daysToExpiration'] < maxDTE) & (df['volumeOpenInterestRatio'] > minVolOIrate)
    selected_df = df[calls]
    return(selected_df)

# import data
df = pd.read_csv('barchart_unusual_activity_2020-06-24.csv')

# apply filters
cherry_df = cherry_pick(df, OutOfMoney = 1.1, maxDTE = 14, minVolOIrate = 3)
selected_options_df = cherry_df[['baseSymbol','baseLastPrice','symbolType','strikePrice','expirationDate','lastPrice', 'volume', 'openInterest',
       'volumeOpenInterestRatio','current_date']]
#selected_options_df.to_csv('options_high_volumeOIratio_20200624.csv')


from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configurations and content
recipients = ['kasperde@hotmail.com']
emaillist = [elem.strip().split(',') for elem in recipients]
msg = MIMEMultipart()
msg['Subject'] = "Unusual option activity"
msg['From'] = 'k.sends.python@gmail.com'

html = """\
<html>
  <head>Some of the most promising soon expiring call options</head>
  <body>
    {0}
  </body>
</html>
""".format(selected_options_df.to_html())

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
