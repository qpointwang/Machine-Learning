import numpy as np
X = np.array([[0, -1], [0, -1], [0, -2], [0, 1], [0, 1], [0, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X, Y) 
print(clf.predict([[-0.8,-1]]))

print(X.shape)
print(Y.shape)

