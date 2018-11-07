# SVM-Handwritten-digit-recognition

### [python库下载地址](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 
### [readme编写方法](https://blog.csdn.net/u012067966/article/details/50736647)

利用支持向量机解决手写数字识别的问题：<br>
首先使用python sklearn.svm库，直接调用库解决问题，然后再看看能都自己编写的SVM代码，解决问题，更好的理解支持向量机。<br>
这里提供的mnist的数据集，mnist_train.mat和mnist_train_labels.mat分别为60000个数据和对应的标签，使用如下代码提取出数据：<br>
```python
import scipy.io as sio
all_data = sio.loadmat('mnist_train.mat')['mnist_train'] #60000*784 array
all_data_lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels'] #60000*1 array
```
由于没有提供测试集，可以将训练集分出一部分做测试集。


#### a2使用50000个训练集，10000个测试集，正确率为0.945  ovr
#### a3使用50000个训练集，10000个测试集，正确率也为0.945  ovo


SimpleSMO.py使用简化版的SMO算法实现手写数字的二分类任务，效率极低，测试代码暂时没有写，因为不知道如何利用b, alphas去测试，得到的记过有错误。并且没有扩展到核函数。


PlattSMO.py使用完整版的SMO算法，据PDF和python机器学习实战书上写着效率会有提高，但是运行到1000个样本时，运行时间就已经很长了。此算法也没有使用核函数，但是书中有提供扩展，我暂时不会做，由于目前的版本只能做到二分类，因此首先想要采用ovo或ovr来实现多分类的情况。


由于SVM是借助二次规划来求解支持向量，而求解二次规划将涉及m阶矩阵的计算（m为样本的个数），当m数目很大时该矩阵的存储和计算将耗费大量的机器内存和运算时间。针对以上问题的主要改进有有J.Platt的SMO算法、T.Joachims的SVM 、C.J.C.Burges等的PCGC、张学工的CSVM以及O.L.Mangasarian等的SOR算法。

b1
b1添加了OVR，一对多的多分类支持向量机方法。目前100个训练集，10000个测试集，正确率为66.74%；200个训练集，10000个测试集，正确率为73.96%,300个训练集，10000个测试集，正确率为75.49%，500个训练集，10000个测试集，正确率为74.70%，1000个训练集，10000个测试集，正确率为74.25%。


OVO为一对一，2000个训练集，正确率为79.08%，20000个训练集，正确率为70.38%。
