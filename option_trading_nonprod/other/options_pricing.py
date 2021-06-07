from math import log, sqrt, pi, exp
import numpy as np
from scipy.stats import norm

def d1(S,K,T,r,sigma):
	return(np.log(S/K)+(r+sigma**2/2.)*T)/sigma*np.sqrt(T)
def d2(S,K,T,r,sigma):
	return d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

def bs_call(S,K,T,r,sigma):
	return S*norm.cdf(d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

def bs_put(S,K,T,r,sigma):
	return K*np.exp(-r*T)-S+bs_call(S,K,T,r,sigma)

def getBSCallPriceWrapper(df, Scol = 'baseLastPrice', Kcol = 'strikePrice', eventDateCol = 'exportedAt', expirationDateCol = 'expirationDate', sigmaCol = 'volatility', riskFree = 0.01):
	S = df[Scol]
	K = df[Kcol]
	T = (pd.to_datetime(df[expirationDateCol]) - pd.to_datetime(df[eventDateCol])).dt.days / 365
	sigma = df[sigmaCol] / 100
	r = riskFree
	return bs_call(S, K, T, r, sigma)