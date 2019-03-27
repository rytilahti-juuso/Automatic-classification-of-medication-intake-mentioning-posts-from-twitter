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
for i in X_train.index:
    X_train.at[i, 'text']= re.sub(r'\W', ' ', str(X_train.at[i, 'text']))
    X_train.at[i, 'text'] = X_train.at[i, 'text'].lower()
    X_train.at[i, 'text'] =  re.sub(r'\s+[a-z]\s+', ' ', X_train.at[i, 'text'])
    X_train.at[i, 'text'] = re.sub(r'^[a-z]\s+', ' ', X_train.at[i, 'text'])
    X_train.at[i, 'text'] = re.sub(r'\s+', ' ', X_train.at[i, 'text'])
