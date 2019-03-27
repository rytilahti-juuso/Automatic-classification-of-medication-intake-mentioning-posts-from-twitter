#import os
#os.chdir(r'C:\Users\The Risk Chief\Documents\GitHub\Automatic-classification-of-medication-intake-mentioning-posts-from-twitter')
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


#Importing data
column_names = ["tweet_id", "name", "train_id", "rank", "text"]
data = pd.read_csv('TW_ST_2/train.txt', delimiter= "\t", names = column_names)
formatted_data = data.loc[:, ["rank", "text"]]

X_train = formatted_data.loc[:,["text",]]
y_train = formatted_data.loc[:,["rank",]]

# Data preprocessing
#def data_preprocessing()    
for i in X_train.index:
    X_train.at[i, 'text']= re.sub(r'\W', ' ', str(X_train.at[i, 'text']))
    X_train.at[i, 'text'] = X_train.at[i, 'text'].lower()
    X_train.at[i, 'text'] =  re.sub(r'\s+[a-z]\s+', ' ', X_train.at[i, 'text'])
    X_train.at[i, 'text'] = re.sub(r'^[a-z]\s+', ' ', X_train.at[i, 'text'])
    X_train.at[i, 'text'] = re.sub(r'\s+', ' ', X_train.at[i, 'text'])

    
# SAME TO DEVEL

#Importing data
column_names = ["tweet_id", "name", "train_id", "rank", "text"]
data = pd.read_csv('TW_ST_2/devel.txt', delimiter= "\t", names = column_names)
formatted_data = data.loc[:, ["rank", "text"]]

X_devel = formatted_data.loc[:,["text",]]
y_devel = formatted_data.loc[:,["rank",]]

# Data preprocessing
#def data_preprocessing()    
for i in X_devel.index:
    X_devel.at[i, 'text']= re.sub(r'\W', ' ', str(X_devel.at[i, 'text']))
    X_devel.at[i, 'text'] = X_devel.at[i, 'text'].lower()
    X_devel.at[i, 'text'] =  re.sub(r'\s+[a-z]\s+', ' ', X_devel.at[i, 'text'])
    X_devel.at[i, 'text'] = re.sub(r'^[a-z]\s+', ' ', X_devel.at[i, 'text'])
    X_devel.at[i, 'text'] = re.sub(r'\s+', ' ', X_devel.at[i, 'text'])

    
# TF_IDF FOR BOTH
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=4000, min_df = 10, max_df = 0.6, stop_words = stopwords.words('english'))
X_train = vectorizer.fit_transform(X_train["text"])


# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Fitting devel to tf-idf model
X_devel = vectorizer.transform(X_devel["text"])
text_rank_pred = classifier.predict(X_devel)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(text_rank_pred, y_devel)

from sklearn.metrics import f1_score
micro_f_score = f1_score(y_devel, text_rank_pred, average='micro')
#micro f-score = 0.6991