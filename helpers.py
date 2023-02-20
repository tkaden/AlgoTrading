import pandas as pd
import numpy as np

r = 0.05 # constant short rate
sigma = 0.2 # constant volatility

def generate_sample_data(rows, cols, freq='1min'):
    rows = int(rows)
    cols = int(cols)
    index = pd.date_range('2023-1-1', periods=rows, freq=freq)
    dt = (index[1] - index[0]) / pd.Timedelta(value='365D')
    columns = ['No%d' % i for i in range(cols)]
    # generate random walk
    raw = np.exp(np.cumsum((r-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((rows, cols)), axis=0))
    # normalize the data to start at 100
    raw = raw / raw[0] * 100
    # generate the dataframe object
    df = pd.DataFrame(raw, index=index, columns=columns)
    return df

