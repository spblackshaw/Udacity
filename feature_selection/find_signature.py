#!/usr/bin/python3

import joblib
import numpy
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "./feature_selection/word_data_overfit.pkl" 
authors_file = "./feature_selection/email_authors_overfit.pkl"
word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### your code goes here
clf = DecisionTreeClassifier().fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

importances = clf.feature_importances_
max_import = [0, 0] # Index value, value
counter = 0
for i in range(len(importances)):
    current = importances[i]
    if current >= 0.2:
        print(current)
        counter += 1
        if current > max_import[1]:
            max_import = i, current
print(max_import, counter)

print(vectorizer.get_feature_names_out()[max_import[0]])

print("here")
