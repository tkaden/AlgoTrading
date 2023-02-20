import pandas as pd
import numpy as np
from scipy.optimize import brute

class SMAVectorBacktestor(object):

    def __init__(self, symbol, SMA1, SMA2, start, end):
        self. symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()
    
    def get_data(self):
        # Prepares the data
        raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        self.data = raw

    def set_parameters(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()
    
    def run_strategy(self):
        data = self.data.dropna().copy()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        self.results = data
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 6), round(operf, 6)

    def plot_results(self):
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol, self.SMA1, self.SMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))

    def update_and_run(self, SMA):
        # Updates the SMA parameters and returns the negative absolute performance of the strategy
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        # Finds global maximum given the SMA parameter ranges
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)

if __name__ == '__main__':
    smabt = SMAVectorBacktestor ('EUR=', 42, 252, '2013-1-1', '2017-12-31')
    print(smabt.run_strategy())
    smabt.set_parameters(SMA1=50, SMA2=200)
    print(smabt.run_strategy())
    print(smabt.optimize_parameters((1, 100, 1), (100, 300, 1)))