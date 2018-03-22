
# coding: utf-8

# In[2]:


import nltk
from nltk.corpus import gutenberg
import tensorflow as tf
import numpy as np


# In[3]:


guten = gutenberg.sents()


# In[4]:


from sklearn.model_selection import train_test_split

gt_train,gt_test = train_test_split(list(guten),test_size=0.2,random_state=1)

gt_dev,gt_test = train_test_split(gt_test,test_size=0.5,random_state=1)


# ## Data clean

# In[5]:


def list_of_words(train):
    seq = []
    for i in range(len(train)):
        sen = [w.lower() for w in train[i] if w not in [':','/',';','|',"''",'``','(',')','-','--','_','"',',','?'] if w.isalpha()]
        seq += sen
    
    return seq


# In[6]:


def sequences_from_list(data):
    length = 5 + 1          # chnage this 5 to 50
    sequences = list()
    for i in range(length, len(data)):
        seq = data[i-length:i]
        line = ' '.join(seq)
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
    return sequences


# In[24]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# ## Neural Language Model

# In[8]:


from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# In[9]:


ls = list_of_words(gt_train[5:6])
print(len(ls))


# In[26]:


sequences = sequences_from_list(ls)


# In[27]:


out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)


# In[11]:


#give unique number to each word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)


# In[12]:


vocab_size = len(tokenizer.word_index) + 1


# In[13]:


# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# In[15]:


model = Sequential()
model.add(Embedding(vocab_size, 4, input_length=seq_length))
model.add(LSTM(3, return_sequences=True))
model.add(LSTM(3))
model.add(Dense(3, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[17]:


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=20, epochs=200)


# In[18]:


# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

