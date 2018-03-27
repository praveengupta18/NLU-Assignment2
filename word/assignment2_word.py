
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import gutenberg
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# In[2]:


files=['austen-emma.txt','austen-persuasion.txt','austen-sense.txt','bible-kjv.txt','blake-poems.txt']
guten = gutenberg.sents(files)

gt_train,gt_test = train_test_split(list(guten),test_size=0.2,random_state=1)

gt_dev,gt_test = train_test_split(gt_test,test_size=0.5,random_state=1)

UNK = '<unknown>'
gt_train[0].append(UNK)


# In[3]:


def list_of_words(data):
    seq = []
    for i in range(len(data)):
        sen = [w.lower() for w in data[i] if w not in [':','/',';','|',"''",'``','(',')','-','--','_','"',',','?'] if w.isalpha()]
        seq += sen
    return seq


# In[70]:


def replace_less_frequent(data):
    unks = set()
    unique = set(data)
    
    uni_cfd = nltk.FreqDist(data)
    for word,freq in uni_cfd.items():
        if freq == 1:
            unks.add(word)
    
    new_data = [w if w not in unks else UNK for w in data]
            
    return new_data


# In[71]:


def lines_from_list(data):
    length = 30 + 1         
    lines = list()
    for i in range(length, int(len(data))):
        seq = data[i-length:i]
        line = ' '.join(seq)
        lines.append(line)
    return lines


# In[38]:


ls = list_of_words(gt_train)
ls = replace_less_frequent(ls)

lines = lines_from_list(ls)

data2 = '\n'.join(lines)
file = open('lines.txt', 'w')
file.write(data2)
file.close()


# In[39]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
lines = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1


# In[75]:


lines = array(lines)
X = lines[:,:-1]
Y = lines[:,-1]
Y = to_categorical(Y, num_classes=vocab_size)

seq_length = X.shape[1]


# ## Train Model

# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=seq_length))
model.add(LSTM(125, return_sequences=True))
model.add(LSTM(125))
model.add(Dense(125, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=400, epochs=45)

model.save('word.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# ## Calculate Perplexity

# In[58]:


def test_data(test,tokenizer):
    sen = [w if w in tokenizer.word_index else UNK for w in test if w not in [':','/',';','|',"''",'``','(',')','-','--','_','"',',','?'] if w.isalpha()]
    return sen


# In[76]:


ls2 = list_of_words(gt_test)
ls2 = test_data(ls2,tokenizer)

lines2 = sequences_from_list(ls2)

lines2 = tokenizer.texts_to_sequences(lines2)

lines2 = array(lines2)
X = lines2[:,:-1]
Y = lines2[:,-1]
Y = to_categorical(Y, num_classes=vocab_size)
seq_length = X.shape[1]

loss = model.evaluate(X,Y,batch_size=400)
print(np.exp(loss))
