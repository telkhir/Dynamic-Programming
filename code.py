# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:18:53 2020

@author: Taoufik.ELKHIRAOUI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Inputs
T, S0, p, u, d, k = 12, 100, 0.5, 1.05, 0.95, 80

values = dict()
sell_now = dict()
sell_after = dict()
strategy = dict()

def recursive_value(t, s):
    if t == T:
        return np.max([s-k, 0]) 
    else:
        return np.max([s-k, p * recursive_value(t+1,u*s) + 
                       (1-p) * recursive_value(t+1,d*s)])

def dynamic_value(t, s):
    if t == T:
        sell_now[t, s] = s-k
        sell_after[t, s] = 0
        values[t, s] = np.max([sell_now[t, s], sell_after[t, s]])
        return values[t, s] 
    else:
        if (t, s) not in values:
            sell_now[t, s] = s-k
            sell_after[t, s] = p * dynamic_value(t+1,u*s) + (1-p) * dynamic_value(t+1,d*s)
            values[t, s] = np.max([sell_now[t, s], sell_after[t, s]]) 
        return values[t, s] 

start_time = time.time()
a = recursive_value(0, S0)
print(a)
print(" recursive : --- %.5s seconds ---" % (time.time() - start_time))


start_time = time.time()
a = dynamic_value(0, S0)
print(a)
print(" dynamic : --- %.5s seconds ---" % (time.time() - start_time))

t_list = [x[0] for x in list(values.keys())]
s_list = [x[1] for x in list(values.keys())]

for t, s in values.keys():
    if sell_now[t, s] >= sell_after[t, s]:
        strategy[t, s] = 1
    else:
        strategy[t, s] = 0

qq = pd.DataFrame({"time": t_list, "price": s_list})
print(qq)

sell_now_df = pd.DataFrame.from_dict(sell_now.items())
sell_now_df[['time', 'price']] = pd.DataFrame(sell_now_df[0].tolist(),
                                              index=sell_now_df.index)
sell_now_df.drop(columns=[0], inplace=True)
sell_now_df.columns = ["sell_now_value","time", "price"]

print(sell_now_df)
col = np.where(np.array(list(strategy.values())) == 1,'r', 'b')
plt.scatter(t_list, s_list, c=col, label=["ff", "ss"])



