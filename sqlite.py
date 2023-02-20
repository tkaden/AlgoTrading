import sqlite3 as sq3
from helpers import generate_sample_data

con = sq3.connect('data/data.sql')

# data = generate_sample_data(1e6, 5, '1min').round(4)
# data.info()

# data.to_sql('data', con)

query = 'SELECT * FROM data WHERE No1 > 105 and No2 < 108'
res = con.execute(query).fetchall()
print(res[:5])
con.close()