# -*- coding: utf-8 -*- 
# 
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import numpy as np

# file = "./MNIST"


trainNum = 50000
testNum = 10000
all_data = sio.loadmat('../data/mnist_train.mat')['mnist_train']
all_data_lables = sio.loadmat('../data/mnist_train_labels.mat')['mnist_train_labels']
data_lables = np.zeros((len(all_data_lables), 10), dtype=np.float)
for i in range(len(all_data_lables)):
	data_lables[i][all_data_lables[i]] = 1.0
all_data = all_data/255.0
train_data = all_data[0:trainNum]
train_data_lables = data_lables[0:trainNum]
test_data = all_data[trainNum:trainNum+testNum]
test_data_lables = data_lables[trainNum:trainNum+testNum]


index = 0
def next_batch(batch_size):
  global index
  if index * batch_size == len(train_data):
    index = 0
  index += 1
  return train_data[(index-1)*batch_size:(index)*batch_size],train_data_lables[(index-1)*batch_size:(index)*batch_size]

#mnist = input_data.read_data_sets(file, one_hot=True)

x = tf.placeholder('float',[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 这里的softmax可以看成是一个激励（activation）函数或者链接（link）函数，
# 把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。
# 因此，给定一张图片，它对于每一个数字的吻合度可以被softmax函数转换成为一个概率值
y = tf.nn.softmax(tf.matmul(x, W) + b) # 首先，我们用tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的，这里x是一个2维张量拥有多个输入。然后再加上b，把和输入到tf.nn.softmax函数里面。
y_ = tf.placeholder('float',[None, 10])


# 为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。
# 其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），
# 然后尽量最小化这个指标。但是，这两种方式是相同的。
# 
# 一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # tf.reduce_sum 计算张量的所有元素的总和求和
# 这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = next_batch(100)
	sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
	# print(sess.run(cross_entropy,feed_dict={x: batch_xs, y_: batch_ys}))


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: test_data, y_: test_data_lables}))
