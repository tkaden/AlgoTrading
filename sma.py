import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True)
raw.info()
data = pd.DataFrame(raw['EUR='])
data.rename(columns={'EUR=': 'price'}, inplace=True)

data['SMA42'] = data['price'].rolling(42).mean()
data['SMA252'] = data['price'].rolling(252).mean()
print(data.tail())

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300

data.plot(title='EUR/USD | 42 & 252 days SMAs', figsize=(10, 6))

data['position'] = np.where(data['SMA42'] > data['SMA252'], 1, -1)

data.dropna(inplace=True)

data['position'].plot(ylim=[-1.1, 1.1], title='Trading position', figsize=(10, 6))

data['returns'] = np.log(data['price'] / data['price'].shift(1))
data['returns'].hist(bins=35, figsize=(10, 6))
data['strategy'] = data['position'].shift(1) * data['returns']
print(data[['returns', 'strategy']].sum())
print(data[['returns', 'strategy']].sum().apply(np.exp))
data[['returns', 'strategy']].cumsum().apply(np.exp).plot(title='EUR/USD | CAGR', figsize=(10, 6))


# Calculate the annualized risk-return statistics for both stock and strategy
data[['returns', 'strategy']].mean() * 252

np.exp(data[['returns', 'strategy']].mean() * 252) -1

# Calculate annualized standard deviation
data[['returns', 'strategy']].std() * 252 ** 0.5

# Define a new column cumret, with the gross performance over time
data['cumret'] = data['strategy'].cumsum().apply(np.exp)

# Define a new column cummax, with the maximum drawdown over time
data['cummax'] = data['cumret'].cummax()

# Plot the two new columns
data[['cumret', 'cummax']].plot(figsize=(10, 6))
plt.show()

# Calculate the maximum drawdown
drawdown = data['drawdown'] = data['cummax'] - data['cumret']
print(drawdown.max())

# When are the differences equal to zero?
temp = drawdown[drawdown == 0]

# Calculate the time between the two dates
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
print(periods)

