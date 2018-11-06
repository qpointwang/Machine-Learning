import scipy.io as sio
import numpy as np
import math

'''
加载训练集数据
'''
def loadDataSet(trainNum,testNum,number):
	dataSet = sio.loadmat('mnist_train.mat')['mnist_train']
	lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
	dataSet = dataSet/255.0
	
	train_data = dataSet[0:trainNum]
	train_data_lables = lables[0:trainNum]
	test_data = dataSet[trainNum:trainNum+testNum]
	test_data_lables = lables[trainNum:trainNum+testNum]

	return train_data, train_data_lables, test_data, test_data_lables

'''
计算类先验概率Pc，条件概率Pxic
以xi等于1计算
'''
def CalProb(train_data, train_data_lable):
	m, n = train_data.shape
	labelSet = np.unique(train_data_lable) #取标签集合
	numLabel = len(labelSet) #标签个数
	
	NumDc = np.zeros(numLabel) #c类的样本有多少个 |Dc|
	mu = np.zeros((numLabel, n)) # 均值
	sigma = np.zeros((numLabel, n)) # 方差

	for i in range(numLabel):
		# np.where(train_data_lable==i)) 从lable中选出值为i的index
		data = train_data[np.where(train_data_lable==i)[0]]
		NumDc[i] = len(data)
		mu[i] = np.mean(data, axis=0)
		sigma[i] = np.std(data, axis=0)

	Pc = NumDc/m #类先验概率
	'''
	NumDc = np.zeros(numLabel) #c类的样本有多少个 |Dc|
	NumDcxi = np.zeros((numLabel, n)) #Dc中第i个样本取值为xi的样本数目 |Dc,xi|
	rou = np.zeros((numLabel, n))	

	for i in range(numLabel):
		for j in range(m):
			if train_data_lable[j] == i:
				NumDc[i] += 1
				NumDcxi[i] += train_data[j]
	u = NumDcxi.T/NumDc #均值
	for i in range(numLabel):
		for j in range(m):
			if train_data_lable[j] == i:
				rou[i] += (train_data[j]-u.T[i])**2
	print(rou[0])
	rou = rou.T/NumDc +(0.01)
	print(rou.T[0])

	Pc = (NumDc)/(m) #类先验概率， P153 7.19
	'''
	return mu, sigma+0.05, Pc


'''
用测试集进行测试
'''
def forecast(test_data, test_data_lables, mu, sigma, Pc):
	m,n = test_data.shape
	right = 0 # 正确个数
	for i in range(m):# 第i个样本
		Pxc = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (test_data[i] - mu)**2 / (2 * sigma**2))
		P = np.sum(np.log(Pxc),axis=1) + np.log(Pc)
		Pmax = np.argmax(P)
		right = right + (Pmax == test_data_lables[i])
		print("第"+str(i)+"个样本："+str(test_data_lables[i])+" 预测值为："+str(Pmax)+"正确率为："+ str((right/(i+1))))
	'''
	m,n = test_data.shape
	labelSet = set([0,1,2,3,4,5,6,7,8,9])
	numLabel = len(labelSet) #标签个数
	right = 0 # 正确个数
	print(Pc)
	for i in range(m):# 第i个样本
		testdata = test_data[i]
		Pc_log = np.log(Pc) # 第i个测试集属于第j个类的 log 概率   每次循环后重新计算
		for j in range(numLabel):#第j个类
			for k in range(n):#第k个属性
				flag = testdata[k]
				Pc_log[j] += np.log((1/(2*math.pi*rou[j][k])**0.5)*math.exp(-((flag-u[j][k])**2)/(2*rou[j][k])))
				# Pc_log[j] += CalLog(flag, Pxic[j][k])
		
		Pmax = np.argmax(Pc_log)
		right = right + (Pmax == test_data_lables[i])
		print("第"+str(i)+"个样本："+str(test_data_lables[i])+" 预测值为："+str(Pmax)+"正确率为："+ str((right/(i+1))))
	'''

if __name__ == "__main__":
	train_data, train_data_lables, test_data, test_data_lables = loadDataSet(50000, 10000, 0)
	mu, sigma, Pc = CalProb(train_data, train_data_lables)
	forecast(test_data, test_data_lables, mu, sigma, Pc)
