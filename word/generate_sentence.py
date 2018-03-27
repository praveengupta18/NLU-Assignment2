
# coding: utf-8

# In[1]:


from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# In[8]:


def generate(model, tokenizer, len2, input_text, n):
    result = list()
    in_text = input_text
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=len2, truncating='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

file = open('lines.txt', 'r')
text = file.read()
file.close()

lines = text.split('\n')
len2 = len(lines[0].split()) - 1

model = load_model('word.h5')

tokenizer = load(open('tokenizer.pkl', 'rb'))

input_text = lines[randint(0,len(lines))]
print('Input text:',input_text)
print('\n')
generated = generate(model, tokenizer, len2, input_text, 10)
print('Generated text: ',generated)

