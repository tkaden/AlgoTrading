import quandl as q
import configparser

config = configparser.ConfigParser()

q.ApiConfig.api_key = open('quandlapikey.txt', 'r').read()

data = q.get('VOL/TSLA', start_date='2018-1-1',
                      end_date='2020-05-01')

data.info()