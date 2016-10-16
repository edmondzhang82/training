
# coding: utf-8

# # Create a RNN model to text generation
# - RNN model at character level
#     - Input: n character previous
#     - Output: next character
#     - Model LSTM
# - Use 'El Quijote' to train the generator
# 

# In[1]:

# Header
#path_base = '/Users/jorge/'
path_base = '/home/jorge/'

path = path_base + 'data/training/keras/'


import numpy as np
import theano


# ## Download data and generate sequences

# In[2]:

#Download quijote from guttenberg project
# wget http://www.gutenberg.org/ebooks/2000.txt.utf-8
    


# In[3]:

text = open(path + "2000.txt.utf-8").read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))

#Dictionaries to convert char to num & num to char
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[4]:

# cut the text in semi-redundant sequences of maxlen characters
# One sentence of length 20 for each 3 characters
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


# ## Train the model

# In[5]:

'''
X: One row by sentence
    in each row a matrix of bool 0/1 of dim length_sentence x num_chars coding the sentence. Dummy variables
y: One row by sentence
    in each row a vector of bool of lengt num_chars with 1 in the next char position
'''

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('X shape: ',X.shape)
print('y shape: ',y.shape)


# In[6]:

# build the model: 2 stacked LSTM
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM


print('Build model 1')
seq_prev_input = Input(shape=(maxlen, len(chars)), name='prev') 
                
# apply forwards LSTM
forwards1 = LSTM(512, return_sequences=True)(seq_prev_input)
dp1 = Dropout(0.25)(forwards1)

forwards2 = LSTM(512, return_sequences=False)(dp1)
dp2 = Dropout(0.5)(forwards2)

output = Dense(len(chars), activation='softmax')(dp2)

model1 = Model(input=seq_prev_input, output=output)

# try using different optimizers and different optimizer configs
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:

#Print the model


# In[7]:

#Fit model
history = model1.fit(X[:600000], y[:600000], batch_size=256, nb_epoch=30,
           validation_data=(X[600000:], y[600000:]))



# In[ ]:




# In[ ]:

#Save model
model_name = 'text_generation_model1'

json_string = model1.to_json()
open(path + 'models/mdl_' + model_name + '.json', 'w').write(json_string)
model1.save_weights(path + 'models/w_' + model_name + '.h5')


# ## Evaluate model

# In[ ]:

# Load model
from keras.models import model_from_json

model_name = 'text_generation_model1'

model1 = model_from_json(open(path + 'models/mdl_' + model_name + '.json').read())
model1.load_weights(path + 'models/w_' + model_name + '.h5')


# In[ ]:

def sample(a, diversity=1.0):
    '''
    helper function to sample an index from a probability array
    - Diversity control the level of randomless
    '''
    a = np.log(a) / diversity
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def generate_text(sentence, diversity, current_model, num_char=400):
    sentence_init = sentence
    generated = ''
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        preds = current_model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    print('\n\nDIVERSITY: ',diversity,'\n')
    print(sentence_init + generated)

sentence = 'mire vuestra merced '
generate_text(sentence, 0.2, model1)
generate_text(sentence, 0.5, model1)
generate_text(sentence, 1,   model1)
generate_text(sentence, 1.2, model1)



sentence = 'de lo que sucedió a don quijote '
generate_text(sentence, 0.2, model1)
generate_text(sentence, 0.5, model1)
generate_text(sentence, 1,   model1)
generate_text(sentence, 1.2, model1)



sentence = 'de alli a poco comenzaron '
generate_text(sentence, 0.2, model1)
generate_text(sentence, 0.5, model1)
generate_text(sentence, 1,   model1)
generate_text(sentence, 1.2, model1)
