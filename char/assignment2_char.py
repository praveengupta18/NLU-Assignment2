
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import gutenberg
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


# In[5]:


file = open('char_train.txt', 'r')
train = file.read()
file.close()

text = train.lower()

text2 = text.split()
text = ' '.join(text2)

l = 10
lines = list()
for i in range(l, len(text)):
    l2 = text[i-l:i+1]
    lines.append(l2)


# In[ ]:


chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))
lines2 = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    lines2.append(encoded_seq)

vocab_size = len(mapping)

lines2 = array(lines2)
X = lines2[:,:-1]
Y = lines2[:,-1]
lines2 = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(lines2)
y = to_categorical(y, num_classes=vocab_size)


# ## Model Train

# In[ ]:


model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=30, verbose=2)

model.save('char.h5')
dump(mapping, open('mapping.pkl', 'wb'))


# ## Calculate Perplexity

# In[ ]:


file = open('char_test.txt','r')
test = file.read()
file.close()

text = train.lower()

text2 = text.split()
text = ' '.join(text2)

l = 10
lines = list()
for i in range(l, len(text)):
    l2 = text[i-l:i+1]
    lines.append(l2)


# In[ ]:


lines2 = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    lines2.append(encoded_seq)

lines2 = array(lines2)
X = lines2[:,:-1]
Y = lines2[:,-1]
lines2 = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(lines2)
y = to_categorical(y, num_classes=vocab_size)


# In[1]:


loss = model.evaluate(X_test,y_test,batch_size=400, verbose=1)
print(np.exp(loss))

