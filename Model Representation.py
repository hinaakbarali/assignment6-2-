#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


# In[2]:


x_train= np.array([1.0,2.0])
y_train = np.array([300.0,500.0])
print(f"x_train ={x_train}")
print(f"y_train ={y_train}")


# In[3]:


print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"num of training examples is:{m}")


# In[5]:


m=len(x_train)
print(f"num of training examples:{m}")


# In[6]:


i=1
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i}),y^({i}))=({x_i}),({y_i})")


# In[7]:


#plot the data
plt.scatter(x_train,y_train,marker='x',c='r')
plt.title("housing price $")
plt.ylabel('price(in 1000s of dollars)')
plt.xlabel('size(1000 sqft)')
plt.show()


# In[8]:


w = 200
b= 100
print(f"w:{w}")
print(f"b:{b}")


# In[15]:


def compute_cost(x,y,w,b):
    m =x.shape[0]
    f_wb = np.zeros(m)
    
    for i in range(m):
      f_wb[i]= w*x[i]+b
    return f_wb


# In[19]:


tmp_f_wb = compute_model_output(x_train,w,b,)
plt.plot(x-train,tmp_f_wb,c='b',label='our prediction')
plt.scatter(x_train,y_train,marker='x',c='r',label='actual value')
plt.title("housing prices")
plt.ylabel('price(in 1000s of dollars)')
plt.xlabel('size(1000 sqft)')
plt.legend()
plt.show()


# In[20]:


w=200
b=100
x_i=1.2
cost_1200sqft=w*x_i+b
print(f"${cost_1200sqft:0f}thousand dollar")


# In[ ]:




