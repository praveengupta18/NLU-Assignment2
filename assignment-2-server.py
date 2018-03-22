
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import gutenberg
import tensorflow as tf
import numpy as np


# In[ ]:


guten = gutenberg.sents()


# In[ ]:


from sklearn.model_selection import train_test_split

gt_train,gt_test = train_test_split(list(guten),test_size=0.2,random_state=1)

gt_dev,gt_test = train_test_split(gt_test,test_size=0.5,random_state=1)


# ## Data clean

# In[ ]:


def list_of_words(train):
    seq = []
    for i in range(len(train)):
        sen = [w.lower() for w in train[i] if w not in [':','/',';','|',"''",'``','(',')','-','--','_','"',',','?'] if w.isalpha()]
        seq += sen
    
    return seq


# In[ ]:


def sequences_from_list(data):
    length = 50 + 1          # chnage this 5 to 50
    sequences = list()
    for i in range(length, len(data)):
        seq = data[i-length:i]
        line = ' '.join(seq)
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
    return sequences


# In[ ]:


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

# In[ ]:


from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


# In[ ]:


ls = list_of_words(gt_train)
print(len(ls))


# In[ ]:


sequences = sequences_from_list(ls)


# In[ ]:


out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)


# In[ ]:


#give unique number to each word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)


# In[ ]:


vocab_size = len(tokenizer.word_index) + 1


# In[ ]:


# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[ ]:


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)


# In[ ]:


# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

