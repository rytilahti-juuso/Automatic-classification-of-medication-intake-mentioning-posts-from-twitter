# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#https://realpython.com/python-keras-text-classification/#defining-a-baseline-model

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

os.chdir(r'C:\Users\The Risk Chief\Documents\GitHub\Automatic-classification-of-medication-intake-mentioning-posts-from-twitter')


# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./TW_ST_2/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='utf-8') as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# prepare text samples and their labels
print('Loading in comments...')

column_names = ["tweet_id", "name", "train_id", "rank", "text"]
data = pd.read_csv('TW_ST_2/train.txt', delimiter= "\t", names = column_names)

formatted_data = data.loc[:, ["rank", "text"]]

sentences = data["text"].fillna("DUMMY_VALUE").values
X_train = formatted_data.loc[:,["text",]].values
y_train = formatted_data.loc[:,["rank",]]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# print("sequences:", sequences); exit()


print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print("median sequence length:", s[len(s) // 2])


# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))


# pad sequences so that we get a N x T matrix
train_padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', train_padded_data.shape)



# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector

# >>> nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
# >>> nonzero_elements / num_words
# 0.6137280267863937

      
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)





print('Building model...')

#model = Sequential()
#model.add(Embedding(1000, 64, input_length=10))


# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = (Flatten())(x)
output = Dense(3, activation='softmax')(x)

model = Model(input_, output)

model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

print('Training model...')
r = model.fit(
  train_padded_data,
  dummy_y,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)


print('Devel preprocessing...')
devel_data = pd.read_csv('TW_ST_2/devel.txt', delimiter= "\t", names = column_names)
y_devel = devel_data['rank']
devel_sentences = devel_data["text"].fillna("DUMMY_VALUE").values
devel_sequences = tokenizer.texts_to_sequences(devel_sentences)
# pad sequences so that we get a N x T matrix
devel_padded_data = pad_sequences(devel_sequences, maxlen=MAX_SEQUENCE_LENGTH)
devel_pred_categorical_y = model.predict(devel_padded_data)
 

devel_pred_y = []
for row in devel_pred_categorical_y:
#    np.amax(row)
    indices = np.where(row == row.max())
#    print(max(row), "index of the row: ", indices[0][0])
    devel_pred_y.append(indices[0][0]+1)

    
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(devel_pred_y, y_devel)

from sklearn.metrics import f1_score
micro_f_score = f1_score(y_devel, devel_pred_y, average='micro')