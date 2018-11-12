# -*- coding: utf-8 -*- 

# from __future__ import print_function
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

with tf.name_scope('input'):
    x = tf.placeholder('float',[None,784],name='x-input')
    y_ = tf.placeholder("float", [None,10],name='y-input')
# 取每次迭代的x的前十个画图
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 第一层隐藏层
with tf.name_scope('first_layer'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([28 * 28, 1200])
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1200])
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1) #tf.nn.relu是激活函数
    tf.summary.histogram('W_fc1',W_fc1)
    tf.summary.histogram('b_fc1',b_fc1)
    tf.summary.histogram('h_fc1',h_fc1)

# 第二层隐藏层
with tf.name_scope('second_layer'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1200, 840])
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([840])
    with tf.name_scope('h_fc2'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    tf.summary.histogram('W_fc2',W_fc2)
    tf.summary.histogram('b_fc2',b_fc2)
    tf.summary.histogram('h_fc2',h_fc2)

# 输出层
with tf.name_scope('output_layer'):
    with tf.name_scope('W_fc3'):
        W_fc3 = weight_variable([840, 10])
    with tf.name_scope('b_fc3'):
        b_fc3 = bias_variable([10])
    with tf.name_scope('h_fc3'):
        h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
    tf.summary.histogram('W_fc3',W_fc3)
    tf.summary.histogram('b_fc3',b_fc3)
    tf.summary.histogram('h_fc3',h_fc3)
# y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
# cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))

# 使用softmax_cross_entropy_with_logits
with tf.name_scope('softmax_cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = h_fc3))
with tf.name_scope('AdamOptimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('equal'):
    correct_prediction = tf.equal(tf.argmax(h_fc3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('softmax_cross_entropy',cross_entropy)
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log", sess.graph)

for i in range(10000):
    batch = mnist.train.next_batch(50)
    # tf.print(b_fc1,[b_fc1])
    if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict={x:batch[0], y_: batch[1]})
        writer.add_summary(summary, i)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
writer.close()
