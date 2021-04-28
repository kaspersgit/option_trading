# Getting various stock level informative values
# company category/branch
# size/balance sheet related
# recommendations
# calender (upcoming earnings)
#t
import pandas as pd
import yfinance as yf

def get_company_info(ticker):
	stock = yf.Ticker(ticker)
	return stock.info

def extract_topics(ticker_object, lookup_date, topics):
	# Get ticker information
	collect_dict = {}

	if 'recommendation' in topics:
		# Get the recommendations on the stock
		recommendations = ticker_object.recommendations
		if recommendations is not None:
			if len(recommendations[recommendations.index < lookup_date]) > 0:
				collect_dict[topics['recommendation']] = recommendations[recommendations.index < lookup_date]['To Grade'].values[-1]

	if 'dividend' in topics:
		# Get the dividends on the ticker_object
		dividends = ticker_object.dividends
		if dividends is not None:
			if len(dividends[dividends.index < lookup_date]) > 0:
				collect_dict[topics['dividend']] = dividends[dividends.index < lookup_date].values[-1]

	if 'general' in topics:
		# Get the general information on the ticker_object
		general = ticker_object.info
		for x in topics['general']:
			if x in general:
				collect_dict[x] = general[x]

	return collect_dict

def enrich_tickers_with_info(df, topics={'recommendation':'last recommendation', 'dividend':'last_dividend', 'general':['sector','industry','sharesOutstanding','fullTimeEmployees','beta','forwardPE','trailingPE']}):
	# Get column names from topic values
	# difficulty here is caused by array in dict
	colnames = []
	for n in topics:
		names = topics[n]
		if isinstance(names, list):
			for ln in names:
				colnames.append(ln)
		else:
			colnames.append(names)

	final_df = pd.DataFrame(columns=colnames)
	unique_tickers = df['baseSymbol'].unique()
	nr_tickers = len(unique_tickers)
	print("Unique tickers: {}".format(nr_tickers))
	for t in range(nr_tickers):
		if t % 50 == 0:
			print('Tickers done: {}'.format(t))
		ticker = unique_tickers[t]
		print(ticker)
		ticker_object = yf.Ticker(ticker)
		unique_exportedAt = df[df['baseSymbol'] == ticker]['exportedAt'].unique()
		print("Unique export dates: {}".format(len(unique_exportedAt)))
		for exDate in unique_exportedAt:
			print(exDate)
			try:
				information = extract_topics(ticker_object, lookup_date=exDate, topics=topics)
			except Exception as e:
				print("Error occurred: {}".format(e))
				information = pd.DataFrame(columns=['exportedAt','baseSymbol'])

			information['exportedAt'] = exDate
			information['baseSymbol'] = ticker

			final_df = final_df.append(information, ignore_index=True)

	return final_df

if __name__ == '__main__':
	ticker = 'OTIS'
	lookup_date = '2020-06-24'

	ticker_object = yf.Ticker(ticker)
	information = extract_topics(ticker_object, lookup_date = lookup_date, topics={'recommendation':'last recommendation', 'dividend':'last_dividend', 'general':['sector','industry','marketCap','beta','forwardPE','trailingPE']})
	print(information)