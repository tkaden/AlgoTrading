import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class BacktestBase(object):
    '''Base class for event-based backtesting of trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    start: str 
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: float
        amount to be invested at the beginning
    ftc: float
        fixed transaction costs per trade
    ptc: float
        proportional transaction costs per trade

    Methods
    =======
    get_data:
        retrieves and prepares the data
    plot_data:
        plots the symbol data of closing price
    get_date_price:
        returns the date and price of the symbol
    print_balance:
        prints out the current balance
    print_net_wealth:
        prints out the current net wealth
    place_buy_order:
        places a buy order
    place_sell_order:
        places a sell order
    close_out:
        closes out the current position
    '''

    def __init__(self, symbol, start, end, amount, ftc=0.0,
                 ptc=0.0, verbose=True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.get_data()
    
    def get_data(self):
        '''Retrieves and prepares the data.
        '''
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def plot_data(self, cols=None):
        '''Plots the symbol data of closing price.
        '''
        if cols is None:
            cols = ['price']
        self.data['price'].plot(figsize=(10, 6), title=self.symbol)

    def get_date_price(self, bar):
        '''Returns the date and price of the symbol.
        '''
        date = str(self.data.index[bar])[:10]
        price = self.data['price'][bar]
        return date, price
    
    def print_balance(self, bar):
        '''Prints out the current balance.
        '''
        date, price = self.get_date_price(bar)
        print(f'{date} | Balance: {self.amount:.2f}')

    def print_net_wealth(self, bar):
        '''Prints out the current net wealth.
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.amount + self.units * price
        print(f'{date} | Net wealth: {net_wealth:.2f}')
    
    def place_buy_order(self, bar, units=None, amount=None):
        '''Places a buy order.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f'{date} | buyings {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        '''Place a sell order
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f'{date} | sellings {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
    
    def close_out(self, bar):
        '''Close out the current position.
        '''
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print('=' * 55)
        print('Final balance: [$] {:.2f}'.format(self.amount))
        perf = ((self.amount - self.initial_amount) / self.initial_amount * 100)
        print('Net Performance: [%] {:.2f}'.format(perf))
        print('Trades Executed: [#] {:.2f}'.format(self.trades))
        print('=' * 55)

if __name__ == '__main__':
    bb = BacktestBase('EUR=', '2016-1-1', '2016-12-31', 100000.0)
    print(bb.data.info())
    print(bb.data.tail())
    bb.plot_data()