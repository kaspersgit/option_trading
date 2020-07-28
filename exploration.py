#%% 
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
scatter_matrix(df_enr[['reachedStrike','priceDiff', 'daysToExpiration', 'midpoint',
       'volumeOpenInterestRatio', 'nrCalls',
       'meanStrikeCall', 'priceDiffPerc']])
plt.show()

# %%
import seaborn as sns
df_regr['maxPricePerc'] = df_regr['maxPrice'] / df_regr['baseLastPrice']
sns.pairplot(df_regr, y_vars="maxPricePerc", x_vars=ex_vars)

# %%
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=train_set['maxPrice'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# %%
# simple plotting 
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)

df_enr['StrikeBin'] = round(df_enr['strikePrice']/100) * 100
df_group = df_enr.groupby('StrikeBin').mean()
df_group.reset_index(inplace=True)
# create some x data and some integers for the y axis
x = df_group['StrikeBin']
y = df_group['reachedStrike']

# plot the data
ax.plot(x,y)

# %%
# Check the test set and split per different grups
test_df = logit_preds[(logit_preds['priceDiffPerc']>1.05) & (logit_preds['baseLastPrice']<1000)]
test_df['price100'] = np.floor(test_df['baseLastPrice'] / 100) * 100
test_df['prediction10'] = np.floor(test_df['prediction'] * 10) / 10
test_split = test_df[['exportedAt','baseSymbol','baseLastPrice','symbolType','expirationDate','reachedStrike','price100','prediction','prediction10','profitStrike','profitStrike110p'
        ]].groupby(['prediction10'
        ]).agg({'baseSymbol':'count', 'reachedStrike':'mean', 'prediction':'mean', 'baseLastPrice':'mean', 'profitStrike':'sum', 'profitStrike110p':'sum'
        }).rename(columns={'baseSymbol':'nrOccurences'
        }).reset_index()
test_split
#logit_preds.groupby('price100').mean()

# %%
# Dive deeper into specific subset
# interesting columns
cols = ['baseSymbol', 'symbolType','baseLastPrice',  'nextBDopen',
       'strikePrice', 'expirationDate', 'daysToExpiration',  'maxPrice', 'minPrice',
       'lastClose', 'high_plus10p', 'low_min10p', 'reachedStrike', 'revenueStrike', 'profitStrike',
       'profitStrikePerc', 'prediction']
test_df[(test_df['prediction10']==0.7)][cols]

# %%
