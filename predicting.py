# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults

# Load model
model = LogitResults.load('/Users/kasper.de-harder/gits/option_trading/modelLogit')

# %%
# Load newest data
df = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-08.csv')


# Adding some additional columns
df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate'
        ]).agg({'baseSymbol':'count', 'strikePrice':'mean'
        }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
        }).reset_index()
df = pd.merge(df,df_symbol, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate'])
df['const'] = 1..0
# Select columns which are model needs as input but leave out the constant
cols = model.params.index

pred = model.predict(df[cols])
df['prediction'] = pred

# %%
threshold = 0.9
buy_advise = df[(df['prediction'] > threshold) & (df['symbolType']=='Call')& (df['daysToExpiration']<10)]
buy_advise = buy_advise[['baseSymbol', 'baseLastPrice', 'prediction']].drop_duplicates(subset='baseSymbol')
# %%
