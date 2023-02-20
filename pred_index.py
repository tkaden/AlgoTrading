import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
print(raw.info())
symbol = 'EUR='
data = pd.DataFrame(raw[symbol])
data.rename(columns={symbol: 'price'}, inplace=True)

lags = 5
cols = []
for lag in range(1, lags+1):
    col = f'lag_{lag}'
    data[col] = data['price'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['price'], rcond=None)[0]
print(reg)

data['pred'] = np.dot(data[cols], reg)
data[['price', 'pred']].plot(figsize=(10, 6))
# plt.show()


data['return'] = np.log(data['price']/data['price'].shift(1))
data.dropna(inplace=True)

cols = []
for lag in range(1, lags + 1):
    col = f'return_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['return'], rcond=None)[0]
print(reg)

data['pred'] = np.dot(data[cols], reg)
data[['return', 'pred']].iloc[lags:].plot(figsize=(10, 6))
# plt.show()

hits = np.sign(data['return'] * data['pred']).value_counts()
print(hits)
print(hits[0]/sum(hits))

reg = np.linalg.lstsq(data[cols], np.sign(data['return']), rcond=None)[0]
print(reg)

data['pred'] = np.sign(np.dot(data[cols], reg))
data['pred'].value_counts()

hits = np.sign(data['return'] * data['pred']).value_counts()
print(hits)
print(hits.values[0]/sum(hits))

print(data.head())

data['strategy'] = data['pred'] * data['return']
print(data[['return', 'strategy']].sum().apply(np.exp))

data[['return', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()