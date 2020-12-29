# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:18:53 2020

@author: Taoufik.ELKHIRAOUI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
import math
import time

# Inputs
T, S0, p, u, d, k = 250, 100, 0.5, 1.05, 0.95, 80

values = dict()
sell_now = dict()
sell_after = dict()
strategy = dict()

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

# recursive impl of the value function
def recursive_value(t, s):
    if t == T:
        return np.max([s-k, 0]) 
    else:
        return np.max([s-k, p * recursive_value(t+1,u*s) + 
                       (1-p) * recursive_value(t+1,d*s)])

# dynamic programming impl of the value function
def dynamic_value(t, s):
    
    if (t, s) not in values:
        if t == T:
            sell_now[t, s] = truncate(s-k, 2)
            sell_after[t, s] = 0
            values[t, s] = np.max([sell_now[t,s], sell_after[t,s]])
        else:
            sell_now[t,s] = truncate(s-k, 2)
            sell_after[t,s] = truncate(p * dynamic_value(t+1, u*s) + (1-p) * dynamic_value(t+1, d*s),  2)
            values[t, s] = np.max([sell_now[t,s], sell_after[t,s]]) 
        
        if sell_now[t, s] >= sell_after[t, s]:
            strategy[t, s] = 1
        else:
            strategy[t, s] = 0
        
    return values[t, s] 

# rerusive is too slow, if T is big
#start_time = time.time()
#a = recursive_value(0, S0)
#print(a)
#print(" recursive : --- %.5s seconds ---" % (time.time() - start_time))

# dynamic is much faster when T is bigger: 35sec when T =250
start_time = time.time()
a = dynamic_value(0, S0)
print(a)
print(" dynamic : --- %.5s seconds ---" % (time.time() - start_time))


# creation of the dataframe containing all the values
sell_now_df = pd.DataFrame.from_dict(sell_now.items())
sell_now_df[['time', 'price']] = pd.DataFrame(sell_now_df[0].tolist(),
                                              index=sell_now_df.index)
sell_now_df.rename(columns = {1:'sell_now'}, inplace = True) 
   
sell_after_df = pd.DataFrame.from_dict(sell_after.items())
sell_after_df[['time', 'price']] = pd.DataFrame(sell_after_df[0].tolist(),
                                              index=sell_after_df.index)
sell_after_df.rename(columns = {1:'sell_after'}, inplace = True) 

df = pd.merge(sell_now_df, sell_after_df, on=['time','price'])

values_df = pd.DataFrame.from_dict(values.items())
values_df[['time', 'price']] = pd.DataFrame(values_df[0].tolist(),
                                              index=values_df.index)
values_df.rename(columns = {1:'value'}, inplace = True) 
df = pd.merge(df, values_df, on=['time','price'])

strategy_df = pd.DataFrame.from_dict(strategy.items())
strategy_df[['time', 'price']] = pd.DataFrame(strategy_df[0].tolist(),
                                              index=strategy_df.index)
strategy_df.rename(columns = {1:'strategy'}, inplace = True) 

df = pd.merge(df, strategy_df, on=['time','price'])

df = df[["time", "price", "sell_now", "sell_after", "value", "strategy"]]  

df.to_csv("data.csv", index=False)
df["strategy"] = df["strategy"].astype(str)

df["value"] = df["value"] + 0.00001 # so that the 0 values could be seen on the plot

# plotting
fig = px.scatter(df, x="time", y="price", color="strategy", # size="value",
                 hover_data=['price', 'sell_now', 'sell_after', 'value']) # 
plot(fig)

#col = np.where(np.array(list(strategy.values())) == 1,'r', 'b')
#plt.scatter(t_list, s_list, c=col)
#fig = plt.figure(figsize=(10,10))
#sns.set_theme()
#sns.scatterplot(data=df, x="time", y="price", hue="strategy", size="value")



