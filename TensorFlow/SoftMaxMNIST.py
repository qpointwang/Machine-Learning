import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import numpy as np

file = "./MNIST"


dataSet = sio.loadmat('mnist_train.mat')['mnist_train']
lables = sio.loadmat('mnist_train_labels.mat')['mnist_train_labels']
# dataSet = np.array(dataSet,dtype=np.int16)
train_data = dataSet[0:50000]
train_data_lables = lables[0:50000]
test_data = dataSet[50000:60000]
test_data_lables = lables[50000:60000]


mnist = input_data.read_data_sets(file, one_hot=True)

print(mnist)

x = tf.placeholder('float',[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# print(x)
y_ = tf.placeholder('float',[None,10])
# print(W)
# print(b)
# print(y)


cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
