#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df= pd.read_csv('all_stocks_5yr.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


NAN= [(c, df[c].isnull().mean()*100) for c in df]
NAN= pd.DataFrame(NAN, columns=['column_name','percentage'])
NAN


# In[7]:


s= df.sort_values(by='date')


# In[8]:


s.head()


# In[9]:


s.reset_index(inplace=True)


# In[10]:


s.head()


# In[11]:


plt.figure(figsize=(10,7))
plt.plot(s['date'],s['close'])


# In[12]:


close_s= s['close']
close_s


# In[13]:


scaler =MinMaxScaler(feature_range=(0,1))
close_s= scaler.fit_transform(np.array(close_s).reshape(-1,1))
close_s


# In[14]:


train_size= int(len(close_s)*.6)
test_size= len(close_s) - train_size
train_data, test_data= close_s[0:train_size,:], close_s[train_size:len(close_s), :1]


# In[15]:


train_data.shape


# In[16]:


test_data.shape


# In[17]:


def create_dataset(dataset, time_step=1):
    dataX, dataY= [], []
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[18]:


time_step= 100
X_train, Y_train= create_dataset(train_data, time_step)
X_test, Y_test= create_dataset(test_data, time_step)


# In[19]:


print(X_train.shape), print(Y_train.shape)


# In[20]:


print(X_test.shape), print(Y_test.shape)


# In[21]:


X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[22]:


X_train


# In[23]:


X_test


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[25]:


model= Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[26]:


model.summary()


# In[27]:


model.fit(X_train, Y_train)


# In[28]:


train_predict= model.predict(X_train)
test_predict= model.predict(X_test)


# In[29]:


train_predict= scaler.inverse_transform(train_predict)
test_predict= scaler.inverse_transform(test_predict)


# In[31]:


import math
from sklearn.metrics import mean_squared_error


# In[33]:


math.sqrt(mean_squared_error(Y_train, train_predict))


# In[34]:


math.sqrt(mean_squared_error(Y_test, test_predict))


# In[35]:


look_back= 100

trainPredictPlot= np.empty_like(close_s)
trainPredictPlot[:,:]= np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]= train_predict

testPredictPlot= np.empty_like(close_s)
testPredictPlot[:,:]= np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(close_s)-1, :]= test_predict

plt.figure(figsize=(10,7))
plt.plot(scaler.inverse_transform(close_s))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[36]:


len(test_data)


# In[37]:


pred_input= test_data[511:].reshape(1, -1)
pred_input.shape


# In[38]:


temp_input= list(pred_input)
temp_input= temp_input[0].tolist()


# In[39]:


temp_input


# In[40]:


lst_output=[]
n_steps=100
i=0

while(i<30):
    if(len(temp_input)>100):
        pred_input= np.array(temp_input[1:])
        print("{} day input {}".format(i, pred_input))
        pred_input= pred_input.reshape(1, -1)
        pred_input= pred_input.reshape((1, n_steps, 1))
        
        yhat= model.predict(pred_input, verbose=0)
        print("{} day input {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input= temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        pred_input= pred_input.reshape((1, n_steps, 1))
        yhat= model.predict(pred_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


# In[41]:


day_new= np.arange(1, 101)
day_pred= np.arange(101, 131)


# In[42]:


len(close_s)

