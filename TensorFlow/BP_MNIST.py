# -*- coding: utf-8 -*- 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

file = "./MNIST"
mnist = input_data.read_data_sets(file, one_hot=True)

# 初始化权重
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #标准差stddev=0.1
  return tf.Variable(initial)
# 初始化偏置项
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder('float',[None,784])
y_ = tf.placeholder("float", [None,10])


# 第一层隐藏层
W_fc1 = weight_variable([28 * 28, 1200])
b_fc1 = bias_variable([1200])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1) #tf.nn.relu是激活函数


# 第二层隐藏层
W_fc2 = weight_variable([1200, 840])
b_fc2 = bias_variable([840])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)


# 输出层
W_fc3 = weight_variable([840, 10])
b_fc3 = bias_variable([10])
h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

# y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
# cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))

# 使用softmax_cross_entropy_with_logits
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = h_fc3))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())


for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})

