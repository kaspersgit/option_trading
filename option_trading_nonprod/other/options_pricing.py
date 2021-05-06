from math import log, sqrt, pi, exp
from scipy.stats import norm

def d1(S,K,T,r,sigma):
	return(log(S/K)+(r+sigma**2/2.)*T)/sigma*sqrt(T)
def d2(S,K,T,r,sigma):
	return d1(S,K,T,r,sigma)-sigma*sqrt(T)

def bs_call(S,K,T,r,sigma):
	return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

def bs_put(S,K,T,r,sigma):
	return K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)