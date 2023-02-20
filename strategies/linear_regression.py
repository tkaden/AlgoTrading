import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LRVectorBacktester(object):
    '''Class for backtesting linear regression-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs per trade

    Methods
    =======
    get_data:
        retrieves and prepares the data
    select_data:
        selects a time period for the data
    prepare_lags:
        prepares the lagged data for the regression
    fit_model:
        implements the regression step
    run_strategy:
        runs the backtest for the momentum-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol    
    '''

    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        '''Retrieves and prepares the data.
        '''
        raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        '''Selects a time period for the data.
        '''
        data = self.data.loc[start:end].copy()
        return data
    
    def prepare_lags(self, start, end):
        '''Prepares the lagged data for the regression.
        '''
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = 'lag_{}'.format(lag)
            data[col] = data['return'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data
    
    def fit_model(self, start, end):
        '''Implements the regression step.
        '''
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols], self.lagged_data['return'], rcond=None)[0]
        self.reg = reg

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        '''Backtests the trading strategy.
        '''
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'].shift(1) * self.results['return']
        # determine when a trade takes place
        trades = self.results['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        self.results['strategy'][trades] -= self.tc
        self.results['creturns'] = self.amount * self.results['return'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * self.results['strategy'].cumsum().apply(np.exp)
        # gross performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        '''Plots the performance of the strategy compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        else:
            title = '{} | TC = {}%'.format(self.symbol, self.tc * 100)
            self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))
            plt.show()

if __name__ == '__main__':
    lrbt = LRVectorBacktester('.SPX', '2010-1-1', '2018-06-29', 10000, 0.0)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31'))
    print(lrbt.run_strategy('2010-1-1', '2015-12-31',
                            '2016-1-1', '2019-12-31'))
    lrbt = LRVectorBacktester('GDX', '2010-1-1', '2019-12-31', 10000, 0.001)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                            '2010-1-1', '2019-12-31', lags=5))
    print(lrbt.run_strategy('2010-1-1', '2016-12-31',
                            '2017-1-1', '2019-12-31', lags=5))
    lrbt.plot_results()