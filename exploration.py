#%% 
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
scatter_matrix(df_enr[['reachedStrike','priceDiff', 'daysToExpiration', 'midpoint',
       'volumeOpenInterestRatio', 'nrOccurences',
       'meanStrikePrice', 'priceDiffPerc']])
plt.show()

# %%
import seaborn as sns
sns.pairplot(df_mature_call, y_vars="reachedStrike", x_vars=['priceDiff', 'daysToExpiration', 'midpoint',
       'volumeOpenInterestRatio', 'nrOccurences',
       'meanStrikePrice', 'priceDiffPerc'])

# %%
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=train_set['strikePrice'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
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
