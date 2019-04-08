# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Embedding
from keras.models import Model
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils

os.chdir(r'C:\Users\The Risk Chief\Documents\GitHub\Automatic-classification-of-medication-intake-mentioning-posts-from-twitter')
MAX_SEQUENCE_LENGTH = 50
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256
EPOCHS = 15


def load_word2vec(EMBEDDING_DIM):
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
    return word2vec

# encode class values as integers
def encode_rank_return_one_hot_encoded(categorical_values):
    encoder = LabelEncoder()
    encoder.fit(categorical_values)
    encoded_categorical_values = encoder.transform(categorical_values)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_categorical_values = np_utils.to_categorical(encoded_categorical_values)
    return dummy_categorical_values


def create_tokenizer(documents):
        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(documents)
        return tokenizer    
    
# convert the sentences (strings) into integers
def text_to_padded_data(documents, tokenizer):
    sequences = tokenizer.texts_to_sequences(documents)
    print("max sequence length:", max(len(s) for s in sequences))
    print("min sequence length:", min(len(s) for s in sequences))
    s = sorted(len(s) for s in sequences)
    print("median sequence length:", s[len(s) // 2])
    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_data
    
def word2index(tokenizer):
    word2idx =tokenizer.word_index 
    print('Found %s unique tokens.' % len(word2idx))
    return word2idx

def main():
    load_word2vec(EMBEDDING_DIM)
    print('Loading in comments...')    
    column_names = ["tweet_id", "name", "train_id", "rank", "text"]
    data = pd.read_csv('TW_ST_2/train.txt', delimiter= "\t", names = column_names)
    documents = data["text"].fillna("DUMMY_VALUE").values
    rank_hot_encoded = encode_rank_return_one_hot_encoded(data["rank"].values)
    tokenizer = create_tokenizer(documents)
    word2idx = word2index(tokenizer)
    text_to_padded_data(documents, tokenizer)
    

    
    
    
main()