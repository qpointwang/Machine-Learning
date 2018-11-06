from numpy import *
import operator
import scipy.io as sio

'''
加载训练集数据
'''
def loadDataSet():
	all_data = sio.loadmat('usps_train.mat')['usps_train']
	all_data_lables = sio.loadmat('usps_train_labels.mat')['usps_train_labels'] 
	# 数据集的大小为4649，每一个数据集包含一个256的行向量，
	all_data_lables = all_data_lables-1
	# printPic(all_data[0])
	# print(all_data_lables[0])
	train_data = all_data[0:3649]
	train_data_lables = all_data_lables[0:3649]
	test_data = all_data[3649:4649]
	test_data_lables = all_data_lables[3649:4649]
	return train_data, train_data_lables, test_data, test_data_lables


'''
创建临时数据集
'''
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	lables = ['A','A','B','B']
	print(group, lables)
	return group, lables

def classify0(inX, dataSet, lables, k):
	dataSetSize = dataSet.shape[0]
	# print(dataSetSize)
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# print(diffMat)
	sqDiffMat = diffMat**2
	# print(sqDiffMat)
	sqDistances = sqDiffMat.sum(axis=1)
	# print(sqDistances)
	distances = sqDistances**0.5
	# print(distances)
	sortedDisIndicies = distances.argsort()
	# print(sortedDisIndicies)
	classCount = {}
	for i in range(k):
		voteIlable = lables[sortedDisIndicies[i]]
		classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
		# print(classCount)
	soertedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return soertedClassCount[0][0]


def printPic(data):
	num = len(data)**0.5
	#print(num)
	for i in range(len(data)):
		if i%num == 0:
			print()
		if data[i] > 0:
			print(0, end='')
		else:
			print(1, end='')


def testDate(train_data, train_data_lables, test_data, test_data_lables):
	correctNum = 0
	for i in range(len(test_data)):
		# printPic(test_data[i])
		result = classify0(test_data[i],train_data,array(mat(train_data_lables).T)[0],3) # k邻近算法的k
		if result == test_data_lables[i]:
			correctNum = correctNum + 1
		print('第'+str(i)+'组，正确应为：'+str(test_data_lables[i][0])+'  结果为：' + str(result) +'  正确率为：'+str(float(correctNum)/(i+1)))
	#print(correctNum)

if __name__ == "__main__":
	train_data, train_data_lables, test_data, test_data_lables = loadDataSet()
	testDate(train_data, train_data_lables, test_data, test_data_lables)

