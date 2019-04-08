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




def main():
    MAX_SEQUENCE_LENGTH = 50
    MAX_VOCAB_SIZE = 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 256
    EPOCHS = 15