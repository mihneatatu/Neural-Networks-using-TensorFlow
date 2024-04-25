#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense, Input
from keras.optimizers import SGD


# ## The Data

# Import the cal_housing_clean.csv file with pandas. Separate it into a training (70%) and testing set(30%).

# In[2]:


df=pd.read_csv('/Users/sandinatatu/Desktop/Tensorflow-Bootcamp-master/02-TensorFlow-Basics/cal_housing_clean.csv')

df.head()


# Separate your features and target data (medianHouseValue) into training and testing sets, with 33% reserved for testing.

# In[3]:


X = df.iloc[:, :6]
y=df['medianHouseValue']

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Scale the Feature Data
# 
# Use sklearn preprocessing to create a MinMaxScaler for the feature data. Fit this scaler only to the training data. Then use it to transform X_test and X_train. Then use the scaled X_test and X_train along with pd.Dataframe to re-create two dataframes of scaled data.

# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Fit a Densely Connected Neural Network to the Training Data
# 

# Construct a Densely Connected Neural Network with 3 hidden layers, each having 6 neurons.

# In[5]:


model = Sequential()
model.add(Input(shape=(6,))) 
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))  


learning_rate = 0.01  
optimizer = SGD(learning_rate=learning_rate)


model.compile(loss='mse',
              optimizer=optimizer)

model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True, verbose=1)


# ## Compute the RMSE on the Test Data

# In[7]:


mse = model.evaluate(X_test, y_test, verbose=0)

print("Root Mean Squared Error on Test Data:", mse**0.5)

