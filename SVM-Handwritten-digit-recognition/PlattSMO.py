#-*- coding: UTF-8 -*-
import numpy as np
import scipy.io as sio
from numpy import *
import logging
#import matplotlib.pyplot as plt
 
 
'''==============================================================================================
完整的SMO SVM算法
'''
 
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


'''
加载训练集数据
'''
def loadDataSet(trainNum,testNum,numberA,numberB):
    all_data = sio.loadmat('mnist_train.mat')['mnist_train']
    all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
    all_data = all_data/255.0
    train_data = all_data[0:trainNum]
    train_data_lables = all_data_lables[0:trainNum]
    test_data = all_data[trainNum:trainNum+testNum]
    test_data_lables = all_data_lables[trainNum:trainNum+testNum]


    labletrain = []
    traindata = []
    for i in range(trainNum):
        if train_data_lables[i][0]==numberA:
            labletrain.append(1)
            traindata.append(train_data[i])
        elif train_data_lables[i][0]==numberB:
            labletrain.append(-1)
            traindata.append(train_data[i])
    alphas = mat(zeros((trainNum,1)))

    labletest = []
    testdata = []
    for i in range(testNum):
        if test_data_lables[i][0]==numberA:
            labletest.append(1)
            testdata.append(test_data[i])
        elif test_data_lables[i][0]==numberB:
            labletest.append(-1)
            testdata.append(test_data[i])

    return mat(traindata), mat(labletrain).transpose(), mat(testdata), mat(labletest), alphas

'''
加载测试数据集
'''
def loadTestDataSet(trainNum,testNum):
    all_data = sio.loadmat('mnist_train.mat')['mnist_train']
    all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
    all_data = all_data/255.0
    test_data = all_data[trainNum:trainNum+testNum]
    test_data_lables = all_data_lables[trainNum:trainNum+testNum]
    return mat(test_data), mat(test_data_lables)

 
def calcEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T * (os.X*os.X[k,:].T) + os.b)
    Ek = fXk - float(os.labelMat[k])
    return Ek
 
'''
函數功能：將輸入的元素限定在一個範圍內
'''
def clipAlpha(input, Low, high):
    if input>high:
        input = high
    if input<Low:
        input = Low
 
    return input
 
'''
函數功能：在輸入的參數i和m之間隨機選擇一個不同於i的數字，也就是在選定了i之後隨機選取一個與之配對的alpha的取值的下標
'''
def selectJrand(i, m):
    j = i
    while j==i:
        j = int(np.random.uniform(0, m))
 
    return j
 
'''
函數功能：選擇一個SMO算法中與外層配對的alpha值的下標
'''
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]  #非零元素的索引号
    if (len(validEcacheList)) > 1:  #啓發式選取配對的j，計算誤差
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:# 如果是第一次循环的话，随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej
 
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
 
'''
SMO算法中的優化部分
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or \
        ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>oS.C)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # print(i,j)
        logging.info('i:'+str(i)+'  j:'+str(j))
        # logging.info(j)
        '''公式(7)'''
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if H == L:
            logging.info('H==L program continued')
            # print("H==L program continued")
            return 0
        # print(L,H)
        logging.info('L:'+str(L)+'  H:'+str(H))
        '''公式（8）（9）'''
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        if 0 <= eta:
            # print("eta>=0 program continued")
            return 0
        logging.info('eta:'+str(eta))
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough %s" % ("program continued"))
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i) #更新誤差緩存'
 
        '''設置常數項 b '''
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i] and oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j] and oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
 
'''
完整版SMO算法
dataMatIn: 訓練數據
classLabels: 數據標籤
C: 常量
toler: 容錯度
maxIter: 最大迭代次數
kTup=('lin', 0): 核函數類型
'''
def SMOP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged>0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                logging.info("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
                print("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0) * (oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                logging.info("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
                print("fullSet, iter: %d    i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (0==alphaPairsChanged):
            entireSet = True
        logging.info("iteration number: %d" % (iter))
        print ("iteration number: %d" % (iter))
    return oS.b, oS.alphas
 
 
'''
函數功能：由計算出來的alphas獲得進行分類的權重向量
alphas: 計算出來的alpha向量
dataArr: 訓練數據
classLabels: 數據標籤
'''
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels)
    rows, cols = np.shape(dataArr)
    w = np.mat(np.zeros((cols, 1)))
    for i in range(rows):
        w += np.multiply(alphas[i, :]*labelMat[:, i], X[i,:].T)
    return w
"""
def printPic(data):
    #print(type(data))
    #print(data.shape)
    data =data.T
    num = len(data)**0.5
    #print(num)
    for i in range(len(data)):
        if i%num == 0:
            print()
        if data[i] > 0:
            print(0, end='')
        else:
            print(1, end='')
"""




def SVM_OVR():
    # 每个分类器参数存储 list
    B = [] #b
    ALPHAS = [] #alphas
    W = [] #w
    # 数字0-9进行遍历，然后对每一个数字进行一对一的训练
    for currentNum in range(10):
        print(currentNum)
        dataMatrix, labelMat, dataMatrix_test, labelMat_test, alphas = loadDataSet(200, 10000, currentNum) #对数字currentNum生成数据集
        b, alphas = SMOP(dataMatrix, labelMat, 1.0, 0.0001, 30) #训练
        w = calcWs(alphas, dataMatrix, labelMat.T) # w的计算
        B.append(b)
        ALPHAS.append(alphas)
        W.append(w)
    return B, ALPHAS, W



def SVM_OVO():
    # 每个分类器参数存储 list
    B = [] #b
    ALPHAS = [] #alphas
    W = [] #w
    num = 0
    ovo = []
    # 数字0-9进行遍历，然后对每一个数字进行一对一的训练
    for currentNum in range(10):
        for nextNum in range(currentNum+1,10):
            print(currentNum,nextNum)
            ovo.append([currentNum,nextNum])
            num += 1
            dataMatrix, labelMat, dataMatrix_test, labelMat_test, alphas = loadDataSet(20000, 10000, currentNum,nextNum) #对数字currentNum生成数据集
            #print(dataMatrix.shape)
            b, alphas = SMOP(dataMatrix, labelMat, 1.0, 0.0001, 30) #训练
            w = calcWs(alphas, dataMatrix, labelMat.T) # w的计算
            B.append(b)
            ALPHAS.append(alphas)
            W.append(w)
    return B, ALPHAS, W, ovo


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log',
                filemode='w')    
    # dataMatrix, labelMat, dataMatrix_test, labelMat_test, alphas = loadDataSet(100, 10000, 0)
    # printPic(dataMatrix_test[0])
    # b, alphas = SMOP(dataMatrix, labelMat, 1.0, 0.0001, 30)
    # w = calcWs(alphas, dataMatrix, labelMat.T)
    # 训练数据 6674
    B, ALPHAS, W, ovo= SVM_OVO()

    #B, ALPHAS, W = SVM_OVR()
    # 加载测试数据集
    dataMatrix_test, labelMat_test = loadTestDataSet(20000, 10000)
    # 结果集
    results = []
    # 测试，对每一个分类器都进行测试，最终的结果便是这四个值中最大的一个
    for i in range(45):
        w = W[i]
        b = B[i]
        result = dataMatrix_test*mat(w)+b
        results.append(result)
    print(results)
    num = 0
    for i in range(10000):
       # temp =[]
        vote = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0}
        for j in range(45):
             #print(float(results[j][i]))
             if ((float(results[j][i])) >= 0):
                 vote[str(ovo[j][0])] = vote[str(ovo[j][0])] + 1
             else:
                 vote[str(ovo[j][1])] = vote[str(ovo[j][1])] + 1
        #print(vote)
        index = max(vote, key=vote.get)
        #print(max(vote, key=vote.get))
         #   temp.append(results[j][i])
        #index = temp.index(max(temp))
        if (int(index) == int(labelMat_test[i])):
            print("----")
            num += 1
        print(int(index),int(labelMat_test[i]))
    print(num)
#"""
# 2000 7908
# 2000 7038
