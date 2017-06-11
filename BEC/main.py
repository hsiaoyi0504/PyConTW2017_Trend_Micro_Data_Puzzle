import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import zipfile

zf = zipfile.ZipFile('./TrendMicro-BEC-dataset-pycon.zip', 'r')
print(zf.namelist())

train_data=pd.read_csv(zf.open('train.csv'))

Y=list(train_data.ix[:, 2])
content=list(train_data.ix[:, 1])

tfidf_vectorizer = TfidfVectorizer(max_features=3000, binary=False, lowercase=True, stop_words='english', sublinear_tf=True)
tfidf = tfidf_vectorizer.fit_transform(content)

clf = LogisticRegression(C=5)
print('Cross-validation:', np.mean(cross_val_score(clf, tfidf, Y)))

clf = clf.fit(tfidf, Y)
print(clf.score(tfidf,Y))

test_data=pd.read_csv(zf.open('test.csv'))
print(test_data)

content=list(test_data.ix[:, 1])
tfidf = tfidf_vectorizer.transform(content)
for i, p in enumerate(clf.predict(tfidf)):
    print(test_data.iloc[i,0], p)