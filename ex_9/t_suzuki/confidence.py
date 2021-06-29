import numpy as np
import pandas as pd
from scipy import stats

acc = pd.Series([293/300, 293/300, 293/300, 293/300, 0.97]) #293/300, 293/300, 292/300, 292/300, 292/300])
alpha = 0.95
d = len(acc)
m = acc.mean()
s = stats.sem(acc)

bottom, up = stats.t.interval(alpha=alpha, df=d, loc=m, scale=s)
print('95% interval: {:.4f} < acc < {:.4f}'.format(bottom, up))
