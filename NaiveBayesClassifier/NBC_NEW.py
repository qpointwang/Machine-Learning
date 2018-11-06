import scipy.io as sio
import numpy as np

'''
加载训练集数据
'''
def loadDataSet(trainNum,testNum,number):
	dataSet = sio.loadmat('mnist_train.mat')['mnist_train']
	lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
	dataSet = np.array(dataSet,dtype=np.int16)
	
	dataSet[dataSet>0] = 1 #将大于0的数替换成1
	
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
	m,n = train_data.shape
	labelSet = np.unique(train_data_lable) #取标签集合
	numLabel = len(labelSet) #标签个数

	NumDc = np.zeros(numLabel) #c类的样本有多少个 |Dc|
	NumDcxi = np.zeros((numLabel, n)) #Dc中第i个样本取值为xi的样本数目 |Dc,xi|
	
	for i in range(numLabel):
		data = train_data[np.where(train_data_lable==i)[0]]
		NumDc[i] = len(data)
		NumDcxi[i] = np.sum(data, axis=0)
	Pxic = (NumDcxi.T + 1)/(NumDc + 2) #条件概率，使用拉普拉斯修正，分母加上属性的种类数，这里是2，  P153 7.20
	Pc = (NumDc + 1)/(m + numLabel) #类先验概率， P153 7.19
	return Pxic.T, Pc

'''
取条件概率的对数，以避免数值下溢
flag为样本第i个属性值(0/1)，p为值为1时对应的概率
'''
def CalLog(flag, p):
	return flag * np.log(p) + (1 - flag) * np.log(1 - p)


'''
用测试集进行测试
'''
def forecast(test_data, test_data_lables, Pxic, Pc):
	m,n = test_data.shape
	labelSet = np.unique(test_data_lables) #取标签集合
	numLabel = len(labelSet) #标签个数
	right = 0 # 正确个数

	for i in range(m):# 第i个样本
		testdata = test_data[i]
		Pc_log = np.log(Pc) # 第i个测试集属于第j个类的 log 概率   每次循环后重新计算
		#for j in range(numLabel):#第j个类
		#	for k in range(n):#第k个属性
		#		flag = testdata[k]
		#		Pc_log[j] += CalLog(flag, Pxic[j][k])
		Pc_log += np.sum(CalLog(testdata,Pxic),axis=1)
		Pmax = np.argmax(Pc_log)
		right = right + (Pmax == test_data_lables[i])
		print("第"+str(i)+"个样本："+str(test_data_lables[i])+" 预测值为："+str(Pmax)+"正确率为："+ str((right/(i+1))))


if __name__ == "__main__":
	train_data, train_data_lables, test_data, test_data_lables = loadDataSet(50000, 10000, 0)
	Pxic, Pc = CalProb(train_data, train_data_lables)
	forecast(test_data, test_data_lables, Pxic, Pc)
