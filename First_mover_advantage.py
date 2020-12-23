# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:39:01 2020

@author: Taoufik.ELKHIRAOUI
"""

import numpy as np
import pandas as pd

# inputs
S0, u, d, p, ro = 200, 1.5, 0.5, 0.5, 0.1


def compute_payments(I_l = 1200, I = 1300, I_h = 2000):
    
    values = np.empty((2,2), dtype=object)
    
    S0_prime = 0.9*S0
     
    # both firms Invest
    both_invest_value = int(round(-I + S0_prime + p * (u*S0_prime/ro) + (1-p) * (d*S0_prime/ro)))
    values[0,0] =  (both_invest_value, both_invest_value)
    
    # firm A wait, firm B Invest
    value_A = int(round((p/(1+ro)) * (-I_h + u*S0_prime + u*S0_prime/ro)))
    value_B = int(round(-I_l + S0 + p * (u*S0_prime/ro) + (1-p) * (d*S0_prime/ro)))
    values[1,0] = (value_A, value_B)
    
    # firm A Invest, firm B Wait
    values[0,1] = (value_B, value_A)
    
    # both firms Wait
    both_wait_value = int(round((p/(1+ro)) * (-I + u*S0_prime + u*S0_prime/ro)))
    values[1,1] = (both_wait_value, both_wait_value)
    
    
    values_df = pd.DataFrame(data=values,
                             index=["A Invest Now", "A Wait time 1"],
                             columns=["B Invest Now", "B Wait time 1"])
    
    return values_df

for I_h in [2000, 1800, 1600, 1400, 1300]:
    df = compute_payments(I_h=I_h)
    print('I_h: ', I_h)
    print(df)
    print("##########")


