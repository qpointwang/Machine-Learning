from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
import scipy.io as sio
import numpy as np

# 数据集
dataSet = sio.loadmat('mnist_train.mat')['mnist_train']
lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
# dataSet = dataSet/255.0

train_data = dataSet[0:50000]
train_data_lables = lables[0:50000]
test_data = dataSet[50000:60000]
test_data_lables = lables[50000:60000]

# 多项分布朴素贝叶斯
mnb = MultinomialNB()
mnb = mnb.fit(train_data, train_data_lables[:,0])
mpredict = mnb.predict(test_data)
print(mpredict)
print(accuracy_score(mpredict, test_data_lables)) #0.8435

# 高斯朴素贝叶斯
gnb = GaussianNB()
gnb = gnb.fit(train_data, train_data_lables[:,0])
gpredict = gnb.predict(test_data)
print(gpredict)
print(accuracy_score(gpredict, test_data_lables)) #0.5623

# 伯努利朴素贝叶斯
bnb = BernoulliNB()
bnb = bnb.fit(train_data, train_data_lables[:,0])
bpredict = bnb.predict(test_data)
print(bpredict)
print(accuracy_score(bpredict, test_data_lables)) #0.8484

