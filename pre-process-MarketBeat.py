from option_trading_nonprod.aws import *
from option_trading_nonprod.process.stock_price_enriching import *

# Temp setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###### import data from S3
# Set source and target for bucket and keys
source_bucket = 'project-option-trading'
source_key = 'raw_data/marketbeat/'
# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))


df = load_from_s3(profile="mrOption", bucket=source_bucket, key_prefix=source_key)
print("Raw imported data shape: {}".format(df.shape))
######

# Delete duplicates
df = df.drop_duplicates(subset=['ticker','exportedAt'], keep='first')
print("After dropping duplicates: {}".format(df.shape))

# make fake expiration date to let function work
virt_daysToExpiration = 21
df['expirationDate'] = (pd.to_datetime(df['dataDate']) + timedelta(days=virt_daysToExpiration)).dt.strftime('%Y-%m-%d')
df.rename(columns={'exportedAt': 'exportedAtTimestamp',
				   'dataDate': 'exportedAt',
				   'ticker': 'baseSymbol'},
		  inplace=True)
df['daysToExpiration'] = virt_daysToExpiration

# filter on only mature options
cutoff_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
print('Cutoff date used: {}'.format(cutoff_date))

df = df[df['expirationDate'] < cutoff_date]
print("After filtering on the cutoff date: {}".format(df.shape))

# Get min max first and last price
# the thrown chunkedencodingerror might be temp solved by https://github.com/SamPom100/UnusualVolumeDetector/issues/22
df_ = df[~df.baseSymbol.isin(['NVCR','UA','ADMA'])]
contracts_prices = getContractPrices(df, startDateCol='exportedAt', endDateCol='expirationDate', type='minmax', use_package='yq')

# Get technical indicators
# Get stock prices from 35 days before export date to calculate them
df['exportedAt'] = pd.to_datetime(df['exportedAt'])
df['start_date'] = df['exportedAt'] - timedelta(days=45)
indicators_df = getContractPrices(df, startDateCol='start_date', endDateCol='exportedAt', type='indicators')

# Put dfs together
df_enr = df_.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])
df_enr = df_enr.merge(indicators_df, on=['baseSymbol','exportedAt'])

# Save enriched df as csv
# x2 if we add technical indicators here
df_enr.to_csv('data/marketbeat_yf_enr_1.csv')