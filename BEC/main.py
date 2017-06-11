import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import zipfile

def calculate_features(content):
    features = []
    word_list = ['money','buy','free','order','earn','make','$','&','cost','discont',
    'thank','me','my','i','you','we','our','he','him','his','she','her','they','them',
    'their', 'interview', 'sorry','only','best','regards', 'call', 'jeff','louise','thor'
    'delainey', 'eric', 'patti', 'peter', 'mark', 'guys', 'peggy', 'brent', 'memo', 'lc', 'need', 'high', 'priority',
    'd', "don", 'tammie', 'room', 'group','tomorrow','today', 'will', 'have', 'that', 'wont', '/', 'hi', 'tammie', 'to',
    'think', 'ask', 're', 'sally', '-', ')', 'time', 'in', 'thanks', 'mike', 'help', 'this', 'great', 'working', 'lunch',
    'always', 'nepco', 'ok', 'last', '?', 'mcconnell', 'lng', 'phone', 'deal', 'let']
    for c in content:
        words = word_tokenize(c.lower())
        feature = []
        feature.append(c.count('\r'))
        for w in word_list:
            feature.append(words.count(w))
        features.append(feature)
    
    return features

zf = zipfile.ZipFile('./TrendMicro-BEC-dataset-pycon.zip', 'r')
print(zf.namelist())

train_data=pd.read_csv(zf.open('train.csv'))

Y=list(train_data.ix[:, 2])
content=list(train_data.ix[:, 1])

features = calculate_features(content)

clf = LogisticRegression()

print('Cross-validation:', np.mean(cross_val_score(clf, features, Y)))
clf = clf.fit(features, Y)
print(clf.score(features,Y))

test_data=pd.read_csv(zf.open('test.csv'))
print(test_data)

content=list(test_data.ix[:, 1])
features = calculate_features(content)
for i, p in enumerate(clf.predict(features)):
    print(test_data.iloc[i,0], p)