import scipy.io as sio
from numpy  import *
from sklearn import svm
all_data = sio.loadmat('mnist_train.mat')['mnist_train']
all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
all_data = all_data/255.0
# 所有像素点都在0~1内
train_data = all_data[0:50000]
train_data_lables = all_data_lables[0:50000]

test_data = all_data[50000:60000]
test_data_lables = all_data_lables[50000:60000]


temp = []
for i in range(len(train_data_lables)):
	#print(train_data_lables[i][0])
	temp.append(train_data_lables[i][0])

temps = []
for i in range(len(test_data_lables)):
	#print(test_data_lables[i][0])
	temps.append(test_data_lables[i][0])

print('start')
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(mat(train_data),array(temp))
print('end')

print(test_data[0])
print(test_data[1])

a = clf.predict(mat(test_data))
#print(array(temp))

num = 0
for i in range(len(test_data)):
	if a[i] == array(temps)[i]:
		num = num + 1

print(num)
print(a)
print(array(temps))
score = clf.score(mat(test_data),array(temps))
print(score)
