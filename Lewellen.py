import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm

# Disclaimer: All errors are ours.
# Authors: Anna Ignashkina, Maksim Nezhelskii
def mult(x, s):
	q=1
	for i in range(1,s+1):
		q=q*(x.shift(i-1)+1)
	return(q)

#df is a file containing value-weighted and equal-weighted NYSE(or NYSE/NASDAQ) index including and excluding dividends
#rf is a file containing 3-months T-bill rate

def lewellen(df, rf):
	#rename columns, format the dates
	rf.columns = ['date', 'rate']
	rf['date'] = pd.to_datetime(rf['date'], format='%Y%m%d')
	rf['date'] = rf['date'].dt.to_period('M')

	#rename columns, format the dates
	df.columns = ['date', 'vwd', 'vwx', 'ewd', 'ewx']
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
	df['date'] = df['date'].dt.to_period('M')

	#merge dataframes
	df = pd.merge(df, rf, on='date', how='inner')

	df['year'] = df.date.dt.year

	# settings for the plot
	ticks_font ='Garamond'
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	for label in ax.get_xticklabels():
	    label.set_fontproperties(ticks_font)

	for label in ax.get_yticklabels():
	    label.set_fontproperties(ticks_font)


	df.dropna(inplace=True)

	#calculate dividend-price ratio using index including and excluding dividedns
	# vwd = (P_{t+1}+D_{t+1})/P_{t} - 1, vwx = P_{t+1}/P_{t} - 1 => D_{t+1}/P_{t+1} can be calculated
	df['dpt'] = ((df['vwd']+1).div(df['vwx']+1) - 1) # scale factor for the pic

	#the same logic for the dividend growth D_{t+1}/D_{t}
	df['dtdt1'] = df['dpt'].div(df['dpt'].shift(1)).div(df['vwx']+1)-1

	df.dropna(inplace=True)
	
	# Calculate (D_{t} + D_{t-1} + D_{t-2}... + D_{t-11})/P_{t} used in Lewellen(2004)
	df['dpt_s']=df['dpt']
	for k in range(1, 12):
		df['dpt_s']+=df['dpt'].shift(k).div(mult(df['vwx'],k))

	#Mean dividend price ratio and its s.d.
	print('D/P ratio mean is', df.dpt_s.mean(), ', and D/P ratio s.d. is', df.dpt_s.std())


	#To replicate the results in the paper, one need to take the log of D/P*100, the same for the index return,
	# so basically regress full % on full %, this seems the only specification which gives the closest to the paper results
	# USE WITH CAUTION since it is just my guess
	df['dpt_s'] = df['dpt_s'].apply(lambda x: math.log(x*100))
	df['vwd'] = (df['vwd']+1).apply(lambda x: (math.log(x)-1)*100)

	df['dpt_sl'] = df['dpt_s'].shift(1)

	#set the time interval
	df = df[(df['year']<1995)&(df['year']>1945)].copy()


	# D/P ratio persistence
	# Perfect match with the results in the paper
	result = sm.formula.ols(formula='dpt_s ~dpt_sl', data=df).fit(cov_type='HC1')
	rho = result.params.values[1]
	df['e_dp']  = (df['dpt_s'] - result.predict(df['dpt_sl']))
	print('AR(1) for D/P ratio')
	print(result.summary())

	# Results for the predictive regression
	result = sm.formula.ols(formula='vwd ~dpt_sl', data=df).fit()
	df['e_r'] = df['vwd'] - result.predict(df['dpt_sl'])
	beta_biased = result.params.values[1]
	alpha = result.params.values[0]
	se_biased = result.bse.values[1]
	print('Predictive regression vwd~ D/P')
	print(result.summary())

	#Correlation between error terms, similar results to the reported in the Lewellen(2004), p.231 footnote
	result = sm.formula.ols(formula='e_r ~e_dp', data=df).fit()
	gamma = result.params.values[1]
	print(result.summary())


	# Regression from p231 (A.4)
	df['mu'] = df['dpt_s'] - df['dpt_sl']
	result = sm.formula.ols(formula='vwd ~dpt_sl+mu', data=df).fit()
	df['v'] = df['ewd'] - result.predict(df[['dpt_sl', 'mu']])
	print(result.summary())

	#Calculation of standard errors
	result = sm.formula.ols(formula='dpt_sl ~e_dp', data=df).fit()
	df['s'] = df['dpt_sl'] - result.predict(df[['e_dp']])

	sigma_s = df['v'].std()/(math.sqrt(len(result.fittedvalues))*(df['s'].std()))
	sigma_s_v2 = df['dpt_s'].var()*(df['e_dp'].var())/(df['e_dp'].var()+(1-rho)**2*(df['dpt_s']).var())
	sigma_s_v2 = math.sqrt(sigma_s_v2)
	T=len(result.fittedvalues)
	sigma_s_v2 = df['v'].std()/(math.sqrt(T)*(sigma_s_v2))

	beta_stamb = beta_biased-gamma*(-4/T)
	beta_adj = beta_biased-gamma*(rho-1)


	print('Biased OLS estimates:', beta_biased, se_biased)
	print('Lewellen beta and s.e.:', beta_adj, sigma_s)
	print('Stambaugh beta:', beta_stamb)

	# Optional, plot of the D/P ratio
	
	#df.plot(x='date', y='dpt_s')
	#plt.show()

		
