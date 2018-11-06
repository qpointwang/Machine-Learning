import scipy.io as sio
from numpy  import *
import time
import datetime

'''
加载训练集数据
'''
def loadDataSet(trainNum,testNum,number):
	all_data = sio.loadmat('mnist_train.mat')['mnist_train']
	all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
	all_data = all_data/255.0
	train_data = all_data[0:trainNum]
	train_data_lables = all_data_lables[0:trainNum]
	test_data = all_data[trainNum:trainNum+testNum]
	test_data_lables = all_data_lables[trainNum:trainNum+testNum]


	labletrain =[]
	for i in range(trainNum):
		if train_data_lables[i][0]==number:
			labletrain.append(-1)
		else:
			labletrain.append(1)
	alphas = mat(zeros((trainNum,1)))

	labletest =[]
	for i in range(testNum):
		if test_data_lables[i][0]==number:
			labletest.append(-1)
		else:
			labletest.append(1)

	return mat(train_data), mat(labletrain).transpose(), mat(test_data), mat(labletest), alphas

'''
随机选择随机数，不等于J
'''
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))  # 一直在挑选随机数j，直到不等于i，随机数的范围在0~m
    return j  # 返回挑选好的随机数


'''
门限函数
'''
def clipAlpha(aj,H,L):  # 最大不能超过H，最小不能低于L
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


'''
简化版的SMO函数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 输入数据，标记，常数C，容错率，最大迭代次数
    dataMatrix = mat(dataMatIn);   # 转换成矩阵
    labelMat = mat(classLabels)#.transpose()  # 转换成矩阵，并转置，标记成为一个列向量，每一行和数据矩阵对应
    m,n = shape(dataMatrix)  # 行，列    
    #print(m,n)
    b = 0;  # 参数b的初始化
    alphas = mat(zeros((m,1)))  # 参数alphas是个list，初始化也是全0，大小等于样本数
    iter = 0  # 当前迭代次数，maxIter是最大迭代次数
    #print(alphas)
    while (iter < maxIter):  # 当超过最大迭代次数，推出
        alphaPairsChanged = 0  # 标记位，记录alpha在该次循环中，有没有优化
        #print(alphaPairsChanged)
        for i in range(m):  # 第i个样本
         #   print(i)
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 第i样本的预测类别
            #print('第'+str(i)+'个样本fXi   '+str(fXi))

            Ei = fXi - float(labelMat[i])#if checksz if an example violates KKT conditions  # 误差 3.5.21
            #print('第'+str(i)+'个样本误差Ei   '+str(Ei))
            #是否可以继续优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择第j个样本
                #print('随机选择的样本为'+str(j))
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  # 样本j的预测类别
                Ej = fXj - float(labelMat[j])  # 误差  3.5.22
                #print('第'+str(i)+'个样本fXj   '+str(fXj))
                #print('第'+str(i)+'个样本误差Ej   '+str(Ej))
                alphaIold = alphas[i].copy();  # 拷贝，分配新的内存
                alphaJold = alphas[j].copy();

                if (labelMat[i] != labelMat[j]): #3.5.25
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else: #3.5.26
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: 
                	#print("L==H"); 
                	continue

                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T #3.5.23

                if eta >= 0: 
                	#print("eta>=0"); 
                	continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta #3.5.20
                alphas[j] = clipAlpha(alphas[j],H,L)  # 门限函数阻止alpha_j的修改量过大  3.5.24

                #如果修改量很微小
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                	#print("j not moving enough"); 
                	continue

                # alpha_i的修改方向相反  #3.5.27
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                # 为两个alpha设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T #3.5.29
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T #3.5.30
                # 3.2.28
                if (0 < alphas[i]) and (C > alphas[i]): 
                	b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                	b = b2
                else: 
                	b = (b1 + b2)/2.0

                #print('b:'+str(b))
                #print()
                # 说明alpha已经发生改变
                alphaPairsChanged += 1
                #print(type(i),type(iter),type(alphaPairsChanged))
                #print("iter: %d i:%d, pairs changed %d") % (iter,i,alphaPairsChanged)

        #如果没有更新，那么继续迭代；如果有更新，那么迭代次数归0，继续优化
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("-------------------iteration number: "+str(iter))

    # 只有当某次优化更新达到了最大迭代次数，这个时候才返回优化之后的alpha和b
    return b,alphas

if __name__ == "__main__":
	dataMatrix, labelMat, dataMatrix_test, labelMat_test, alphas = loadDataSet(1000, 100, 0)
	start = datetime.datetime.now()
	b, alphas = smoSimple(dataMatrix, labelMat, 1.0, 0.0001, 30)
	print(b)
	print(alphas)
	end = datetime.datetime.now()
	print((end-start).total_seconds())

	w = multiply(alphas,labelMat_test)
	for i in range(1000):
		fX = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
		print(fX)