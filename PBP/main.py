import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

RANDOM_STATE = 1200
SELECT_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]

train_data=pd.read_csv('PBP-train.csv', header=None)
train_data.fillna(-1, inplace=True)
label_encoder_1 = LabelEncoder()
label_encoder_1.fit(train_data[1])
train_data[1] = label_encoder_1.transform(train_data[1])
label_encoder_2 = LabelEncoder()
label_encoder_2.fit(train_data[2])
train_data[2] = label_encoder_2.transform(train_data[2])
label_encoder_3 = LabelEncoder()
label_encoder_3.fit(train_data[6])
train_data[6] = label_encoder_3.transform(train_data[6])
y=np.array(train_data[0])
X=np.array(train_data.iloc[:, SELECT_INDEX])

test_data=pd.read_csv('PBP-test.csv', header=None)
test_data.fillna(-1, inplace=True)
test_data[1] = label_encoder_1.transform(test_data[1])
test_data[2] = label_encoder_2.transform(test_data[2])
test_data[6] = label_encoder_3.transform(test_data[6])
X_test=np.array(test_data.iloc[:, SELECT_INDEX])

clf = RandomForestClassifier(n_estimators=350, min_samples_split=16, max_depth=17, n_jobs=-1, random_state=RANDOM_STATE)

print('Cross-validation:', np.mean(cross_val_score(clf, X, y)))
clf = clf.fit(X, y)
print('Training:', clf.score(X, y))
y_test = clf.predict(X_test)
print('Prediction:', y_test)