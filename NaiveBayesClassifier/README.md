# NaiveBayesClassifier
贝叶斯分类器

使用拉普拉斯修正，《机器学习》P153 7.19 7.20,并取对数，将连乘转换成连加

NBC.py 60000条数据，前1000做训练，100做测试，正确率0.73;
前50000做训练，10000做测试，正确率0.8484.但是这个方法使用了大量的for循环，算法时间复杂度很高，而NBC_NEW.py采用了numpy一些专有的函数，极大的提高了效率，结果仍然一样。


NaiveBayes_sklearn.py使用sklearn中的三种朴素贝叶斯方法对手写数字进行识别，识别结果分别为：多项分布朴素贝叶斯，0.8435；高斯朴素贝叶斯，0.5623；伯努利朴素贝叶斯，0.8484。

sklearn_test.py 为sklearn的测试用例，开始不知道fit的输入是什么。

NBC_C.py 使用高斯贝叶斯进行计算，讲每个样本的属性值看做连续值，使用书7.18公式。先求均值和方差。每个样本总会出现某一属性值都为0的情况，这时，均值和方差都为零，不再适合正太分布公式，因此这里将方差计算出来以后，整体都加0.05，使得方差不为零。使用目前的方法效率很高，原来注释掉的方法效率很低。0.05时，正确率为0.7429
