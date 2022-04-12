#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import pandas as pd
import numpy as np
import os


# In[3]:


os.chdir(r"D:\project")


# In[4]:


df = pd.read_excel("scu_zoo_0211_최종.xlsx")


# In[5]:


df.dropna(inplace = True)


# In[6]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[7]:


df


# In[9]:


y


# In[10]:


df


# In[11]:


X = df.iloc[:,:19]
y = df['PHY']


# In[12]:


X


# In[13]:


X_train


# In[14]:


X_train.shape


# In[15]:


minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X)
X = minmax_scaler.transform(X)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)


# In[17]:


X_tr, X_val, y_tr, y_val = train_test_split(
     X_train, y_train, test_size=0.2, random_state=42)


# In[24]:


model = Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (19,)))
model.add(layers.Dense(16, activation = 'relu', input_shape = (19,)))
model.add(layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])


# In[26]:


y


# In[28]:


hist = model.fit(X_tr, y_tr, epochs=10000, batch_size=32, shuffle=False, verbose=1,validation_data=(X_val,y_val))


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


history_dict = hist.history
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


# In[31]:


from sklearn.metrics import mean_squared_error, r2_score
y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_tr)
mse_test = mean_squared_error(y_test, y_test_predict)
r2_score_test = r2_score(y_test, y_test_predict)

mse_train = mean_squared_error(y_tr, y_train_predict)
r2_score_train = r2_score(y_tr, y_train_predict)
print(mse_test, r2_score_test)
print(mse_train, r2_score_train)


# In[32]:


import matplotlib.patches as patches


# In[33]:


fig, ax = plt.subplots(figsize=(6, 6))
plt.plot([-0.5, 16], [-0.5, 16], 'black', linestyle = 'dashed', linewidth = 0.5, zorder=10)
ax = plt.gca()
ax.add_patch(patches.Rectangle((-1, -1), 16,  16, edgecolor = None, facecolor = '#Ffdbdb', fill = True))
ax.add_patch(patches.Rectangle((-1, -1), np.log1p(1000000)+1,  np.log1p(1000000)+1, edgecolor = None, facecolor = "#Ffeddb", fill = True))
ax.add_patch(patches.Rectangle((-1, -1), np.log1p(10000)+1,  np.log1p(10000)+1, edgecolor = None, facecolor = "#Fff6db", fill = True))
ax.add_patch(patches.Rectangle((-1, -1), np.log1p(10),  np.log1p(10), edgecolor = None, facecolor = "white", fill = True))
ax.scatter(y_tr, y_train_predict, color = 'purple', marker = "^", alpha = 0.7, s = 100, edgecolors = "k", zorder = 6, label = "Valibration") ### validataion
ax.scatter(y_test, y_test_predict, color = 'red', marker = "o", alpha = 0.7, s = 100, edgecolors = "k", zorder = 7, label = "Test") ### validataion
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth = 0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


# In[209]:


ax.scatter(y_tr, y_train_predict, color = 'dodgerblue', marker = "s", alpha = 0.7, s = 5, edgecolors = "k", zorder = 5) ## training


# In[163]:


X_tr


# In[ ]:




