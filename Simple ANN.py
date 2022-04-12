#!/usr/bin/env python
# coding: utf-8

# 1. Load packages

# In[1]:


# Simple ANN for dam gate operation status prediction
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 2. Load water quality data

# In[16]:


os.chdir(r"D:\")


# In[17]:


df = pd.read_excel("Sample data.xlsx")


# In[18]:


df.dropna(inplace = True)


# In[20]:


X = df.iloc[:,1:15] #independent var
y = df['Weir'] # dependent var Dam gate operation status(0 is closed, 1 is opened)


# In[21]:


# Scale independent var
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X)
X = minmax_scaler.transform(X)


# In[22]:


# Split test data
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)


# In[23]:


# Split train and validation data
X_tr, X_val, y_tr, y_val = train_test_split(
     X_train, y_train, test_size=0.2, random_state=42)


# 3. Simple ANN structure

# In[24]:


model = Sequential() 
model.add(layers.Dense(16, activation = 'relu', input_shape = (14,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# 4. Training

# In[25]:


hist1 = model.fit(X_tr, y_tr, epochs=100, batch_size=64, shuffle=False, verbose=1,validation_data=(X_val,y_val))


# 5. Visualize train and validation loss

# In[26]:


import matplotlib.pyplot as plt


# In[27]:


# No difference Train and validation loss show that model is fitted properly
history_dict = hist1.history
print(history_dict.keys())

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epoch = range(1,len(loss)+1)

plt.plot(epoch,loss,'bo',label='Training Loss')
plt.plot(epoch,val_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss Visualization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 6. Evaluation

# In[51]:


from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_tr)


# In[52]:


pred_X_test = []
for i in model.predict(X_test):
    if i < 0.5:
        pred_X_test.append(0)
    else:
        pred_X_test.append(1) 


# In[55]:


pred_X_train = []
for i in model.predict(X_tr):
    if i < 0.5:
        pred_X_train.append(0)
    else:
        pred_X_train.append(1)


# In[61]:


pred_X_test=np.array(pred_X_test)
pred_X_test=pred_X_test.reshape(-1,1)


# In[62]:


pred_X_train=np.array(pred_X_train)
pred_X_train=pred_X_train.reshape(-1,1)


# In[63]:


# Check test and train accurcy
test_acc = accuracy_score(y_test, pred_X_test)
train_acc = accuracy_score(y_tr, pred_X_train)
print('test acc: '+ str(test_acc))
print('train acc: '+str(train_acc))

