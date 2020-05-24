import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as stats
import math
#%matplotlib inline




dataset = pd.read_csv('./UVXY VIX DEC 2019-subset_v02.csv')

uvxy_iv =  dataset.loc[dataset['symbol'] == 'UVXY']['iv']
vix_iv =  dataset.loc[dataset['symbol'] == 'VIX']['iv']


uvxyc_iv = dataset.loc[(dataset['symbol'] == 'UVXY') & (dataset['call/put']=='C')]['iv']
uvxyp_iv = dataset.loc[(dataset['symbol'] == 'UVXY') & (dataset['call/put']=='P')]['iv']

vixc_iv = dataset.loc[(dataset['symbol'] == 'VIX') & (dataset['call/put']=='C')]['iv']   #.to_frame()
vixp_iv = dataset.loc[(dataset['symbol'] == 'VIX') & (dataset['call/put']=='P')]['iv']   #.to_frame()


uvxyc = dataset.loc[(dataset['symbol'] == 'UVXY') & (dataset['call/put']=='C')]
uvxyp = dataset.loc[(dataset['symbol'] == 'UVXY') & (dataset['call/put']=='P')]
vixc = dataset.loc[(dataset['symbol'] == 'VIX') & (dataset['call/put']=='C')]
vixp = dataset.loc[(dataset['symbol'] == 'VIX') & (dataset['call/put']=='P')]

# mu = np.mean(vixc_iv)
# variance = np.var(vixc_iv)
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
# mu2 = np.mean(vixp_iv)
# variance2 = np.var(vixp_iv)
# sigma2 = math.sqrt(variance2)
# x2 = np.linspace(mu2 - 3*sigma2, mu2 + 3*sigma2, 100)
# plt.plot(x2, stats.norm.pdf(x2, mu2, sigma2), color='red')
# plt.show()
# vixc_iv.reset_index(drop=True, inplace=True)
# vixp_iv.reset_index(drop=True, inplace=True)
# vix_iv.reset_index(drop=True, inplace=True)
# vixdf = pd.concat([vixc_iv,vixp_iv,vix_iv], axis=1)
# vixdf.to_csv('vix.csv', encoding='utf-8')

#print(np.var(uvxyp_iv))
# vixdf = pd.concat([vix_iv,vix_c,vix_p], axis=1)

X = vixp[['adjusted close','log(strike)', 'log(ask)', 'log(bid)', 'log(mean price)', 'log(volume)', 'log(open interest)', 'stock price for iv', 'delta', 'vega', 'gamma', 'theta', 'rho']].values
y = vixp_iv.values



regressor = LinearRegression()
regressor.fit(X, y)

coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
print(coeff_df)
