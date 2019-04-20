#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy.io as sio
import numpy as np
import logging
# 数据集
dataSet = sio.loadmat('mnist_train.mat')['mnist_train']
lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
print(len(dataSet))
dataSet = dataSet/255.0

logging.info("Loading Files...")
train_data = dataSet[0:50000]
train_data_lables = lables[0:50000]
test_data = dataSet[50000:60000]
test_data_lables = lables[50000:60000]
logging.info("Traning...")
clf = svm.SVC(C=1.0, decision_function_shape='ovr')
clf.fit(train_data, train_data_lables[:,0])
logging.info("Training End")
result = clf.predict(test_data)
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='SVM_SKLEARN.log',
                filemode='w')

logging.info(result)
logging.info(accuracy_score(result, test_data_lables))
logging.info(clf.score(test_data, test_data_lables))
