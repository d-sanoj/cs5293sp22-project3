# Importing necessary libraries for the code execution

import pandas
import nltk
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import sys
import os
import wget
import warnings

warnings.filterwarnings("ignore")

nltk.download('stopwords',  quiet=True)

print('Checking and removing the old unredactor.tsv File')
if os.path.exists("unredactor.tsv"):
  os.remove("unredactor.tsv")

print('Downloading unredactor.tsv file')
# Downloading the unredactor.tsv if not existing in the directory
if not os.path.exists('unredactor.tsv'):
    wget.download('https://raw.githubusercontent.com/sanojdoddapaneni/cs5293sp22/main/unredactor.tsv')

print('\nExecuting the code, wait for 5 - 10 minutes\n')
# Reading and creating a dataframe from the tsv file
df = pandas.read_csv("unredactor.tsv",on_bad_lines='skip', sep='\t')
df.columns = ['waste','Data','Name','Sentence']
df_data = df[['waste','Data','Name','Sentence']]

# Cleaning the data using several functions
# Source - https://machinelearningknowledge.ai/11-techniques-of-text-preprocessing-using-nltk-in-python/
# Whitespaces removal
def remove_whitespace(text):
    return  " ".join(text.split())

df_data1=df_data['Sentence'].apply(remove_whitespace)

df_data2=df_data1.apply(lambda X: word_tokenize(X))

# Stop words removal
en_stopwords = stopwords.words('english')
def remove_stopwords(text):
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)           
    return result

df_data3=df_data2.apply(remove_stopwords)

# Punctuations removal
def remove_punct(text): 
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

df_data4=df_data3.apply(remove_punct)

# Stemming process
def stemming(text):
    porter = PorterStemmer()
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

df_data5=df_data4.apply(stemming)

# Adding the cleaned data to the dataframe
finaldf = pandas.concat([df_data, df_data5], axis=1, join='inner')
finaldf.columns = ['waste','Data','Name','Sentence','Cleaned']
finaldf['CleanedString'] = finaldf.Cleaned.apply(' '.join)

# Dividing the data according to their data labels
training=finaldf[(finaldf['Data']=='training')|(finaldf['Data']=='validation')]
testing = finaldf[finaldf['Data']=='testing']

vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(training['CleanedString']).toarray()
test = vectorizer.transform(testing['CleanedString']).toarray()

model =DecisionTreeClassifier()
model.fit(train, training['Name'])
prediction = model.predict(test)
print(prediction)
print("Precision Score: ", precision_score(testing['Name'], prediction, average = 'weighted', zero_division=1))
print("F1 Score:        ", f1_score(testing['Name'], prediction, average = 'weighted', zero_division=1))
print("Recall Score:    ", recall_score(testing['Name'], prediction, average = 'weighted', zero_division=1))
