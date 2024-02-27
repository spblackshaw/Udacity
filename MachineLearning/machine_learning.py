from sklearn.naive_bayes import GaussianNB
import numpy as np


clf = GaussianNB()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]), "test")
