import os
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

x = np.linspace(0, 10)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
set_seeds()

y = x + np.random.standard_normal(len(x))

reg = np.polyfit(x, y, deg=1)
print(reg)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
plt.plot(x, np.polyval(reg, x), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)
plt.show()