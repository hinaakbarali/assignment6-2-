#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition,plt_stationary,plt_update_onclick
plt.style.use('./deeplearning.mplstyle')


# In[4]:


x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])


# In[7]:


def compute_cost(x,y,w,b):
    Args:
        x(ndarray(m,)):Data,m examples
        y(ndarray(m,)):target values
        w,b(scalar)   : model parameter
    returns
        total_cost(float):the cost of using w,b as the parameters for lir to fit the data points in x and y
    m =x.shape[0]
    
    cost_sum = 0
    for i in range(m)
        f_wb = w* x[i]+b
        cost =(f_wb-y[i])**2
        cost_sum = cost_sum +cost
    total_cost = (1/(2*m))*cost_sum
    
    return total_cost


# In[ ]:




