import matplotlib.pyplot as plt
import pandas as pd

def PredictionVsStrikeIncrease(df, ReachedStrike, notReachedStrike, savefig=False, saveFileName='test.png'):
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.scatter(ReachedStrike['strikePricePerc'], ReachedStrike['prob'], s = 4, color='g', alpha=0.7, label='Did reach strike')
	ax.scatter(notReachedStrike['strikePricePerc'], notReachedStrike['prob'], s = 4, color='r', alpha=0.7,
			   label='Not reached strike')
	ax.legend(loc="upper right")
	ax.set_xlabel('Strike price increase')
	ax.set_ylabel('Predicted probability')
	ax.set_title('All Call options plotted')
	plt.show()
	if savefig:
		fig.savefig(saveFileName)
		print(f'Created and saved scatter plot of strike price increase vs predicted probability')

def GroupsPerformanceComparisonBar(df, high_prob_df, high_prof_df, savefig=False, saveFileName='test.png'):

	df_=df.copy()

	# Create bar plots showing share of successes per strike price increase bucket
	all_strikeIncreaseBin = df_[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
	all_binPercentage = all_strikeIncreaseBin.groupby(level=0).apply(lambda x:
																	 100 * x / float(x.sum())).reset_index(drop=False)

	hprob_strikeIncreaseBin = high_prob_df[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
	hprob_binPercentage = hprob_strikeIncreaseBin.groupby(level=0).apply(lambda x:
																		 100 * x / float(x.sum())).reset_index(drop=False)

	hprof_strikeIncreaseBin = high_prof_df[['baseSymbol','strikePricePercBin','reachedStrikePrice']].groupby(['strikePricePercBin','reachedStrikePrice']).count()
	hprof_binPercentage = hprof_strikeIncreaseBin.groupby(level=0).apply(lambda x:
																		 100 * x / float(x.sum())).reset_index(drop=False)


	fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
	# All contracts
	axs[0].bar(all_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
	if len(all_binPercentage[all_binPercentage['reachedStrikePrice']==1]['baseSymbol']) > 0:
		axs[0].bar(all_binPercentage['strikePricePercBin'].unique(), all_binPercentage[all_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
	axs[0].tick_params('x', labelrotation=45)
	axs[0].title.set_text('All Calls')
	axs[0].set_ylabel('Fraction reaching strike price')

	# high probability contracts
	axs[1].bar(hprob_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
	if len(hprob_binPercentage[hprob_binPercentage['reachedStrikePrice']==1]['baseSymbol']) > 0:
		axs[1].bar(hprob_binPercentage['strikePricePercBin'].unique(), hprob_binPercentage[hprob_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
	axs[1].tick_params('x', labelrotation=45)
	axs[1].title.set_text('High probability')
	axs[1].set_xlabel("Strike price increase with respect to stock price")

	# high probability contracts
	axs[2].bar(hprof_binPercentage.dropna()['strikePricePercBin'].unique(), 100, color='red')
	if len(hprof_binPercentage[hprof_binPercentage['reachedStrikePrice']==1]['baseSymbol']) > 0:
		axs[2].bar(hprof_binPercentage['strikePricePercBin'].unique(), hprof_binPercentage[hprof_binPercentage['reachedStrikePrice']==1]['baseSymbol'], color='green')
	axs[2].tick_params('x', labelrotation=45)
	axs[2].title.set_text('High profitability')

	fig.tight_layout(rect=[0,0,0.9,0.9])
	if savefig:
		fig.savefig(saveFileName)
		print(f'Created and saved bar plot of success ratio all options vs selected options')

def ExpvsActualProfitabilityScatter(df,high_prob_df ,high_prof_df, actualCol, savefig=False, saveFileName='test.png'):
	typeOfActual = 'Max' if actualCol.startswith('max') else 'Actual'

	# Create scatter plot (strike price progress vs predicted probability)
	# rows not appearing in any of the email tables
	not_email_df = df[(~df.index.isin(high_prob_df.index)) & (~df.index.isin(high_prof_df.index))]

	fig = plt.figure()
	# cm = plt.cm.get_cmap('Blues')
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.scatter(not_email_df[actualCol], not_email_df['expPercIncrease'], s=7, color='r', alpha=0.7, label='Not in email')
	ax.scatter(high_prob_df[actualCol], high_prob_df['expPercIncrease'], s=7, color='g', alpha=0.7, label='High probability')
	ax.scatter(high_prof_df[actualCol], high_prof_df['expPercIncrease'], s=7, color='b', alpha=0.7, label='High profitability')
	ax.legend(loc="upper left")
	ax.set_xlabel(f'{typeOfActual} profit')
	ax.set_ylabel('Expected profit')
	ax.set_title(f'Expected vs {typeOfActual} profitability')
	plt.show()
	if savefig:
		fig.savefig(saveFileName)
		print(f'Created and saved scatter plot (expected vs {typeOfActual} profitability)')

def AddDaysFromStartToEnd(df, startCol = 'exportedAt', endCol = 'strikePriceDate'):
	df_ = df.copy()

	# extract nr of days between start and end (if end exist)
	df_['duration'] = (pd.to_datetime(df_[endCol]) - pd.to_datetime(df_[startCol])).dt.days

	return(df_)

def getDaysToStrikeAsShare(df):
	df_ = df.copy()

	# Get time duration from extraction to reaching strike price
	df_ = AddDaysFromStartToEnd(df_, startCol = 'exportedAt', endCol = 'strikePriceDate')
	df_['counting'] = 1

	d2e = df_[['daysToExpiration','counting']].groupby('daysToExpiration').count()
	d2e['activeOptions'] = d2e['counting'][::-1].cumsum()
	d2e = d2e.reset_index(drop=False)
	d2e['daysToExpiration'] = d2e['daysToExpiration'].astype('int')
	d2e = d2e[['daysToExpiration','activeOptions']]

	d2s = df_[['duration','counting']].groupby('duration').count()
	d2s = d2s.reset_index(drop=False)
	d2s.rename(columns={'counting':'reachedStrike'}, inplace=True)
	d2s['duration'] = d2s['duration'].astype('int')

	df_merge = pd.merge_asof(d2s, d2e, left_on=['duration'], right_on=['daysToExpiration'], direction='forward')
	df_merge['strikeReachedShare'] = df_merge['reachedStrike'] / df_merge['activeOptions']

	return df_merge