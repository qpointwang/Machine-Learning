from numpy import *
import operator
import scipy.io as sio

'''
加载训练集数据
'''
def loadDataSet():
	all_data = sio.loadmat('mnist_train.mat')['mnist_train']
	all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
	all_data = all_data/255.0
	train_data = all_data[0:50000]
	train_data_lables = all_data_lables[0:50000]
	test_data = all_data[50000:60000]
	test_data_lables = all_data_lables[50000:60000]
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
	for i in range(len(data)):
		if i%num==0:
			print()
		if data[i] == 0:
			print(0, end='')
		else:
			print(1, end='')


def testDate(train_data, train_data_lables, test_data, test_data_lables):
	correctNum = 0
	for i in range(len(test_data)):
		# printPic(test_data[i])
		result = classify0(test_data[i],train_data,array(mat(train_data_lables).T)[0],50) # k邻近算法的k
		# print('')
		# print(result)
		if result == test_data_lables[i]:
			correctNum = correctNum + 1
		print('第'+str(i)+'组，正确应为：'+str(test_data_lables[i][0])+'  结果为：' + str(result) +'  正确率为：'+str(float(correctNum)/(i+1)))
	print(correctNum)

if __name__ == "__main__":
	# 测试所用的测试代码
	# group, lables = createDataSet()
	# classify0([1.0,1.2], group, lables, 3)
	train_data, train_data_lables, test_data, test_data_lables = loadDataSet()
	testDate(train_data, train_data_lables, test_data, test_data_lables)
	# print(test_data_lables[0])
	#classify0(test_data[0],train_data,array(mat(train_data_lables).T)[0],50)
	#printPic(test_data[0])
	#for i 
