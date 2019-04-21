import tensorflow as tf
import numpy as np

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))  # 常量可以直接输出，而变量需要初始化
a = tf.constant(10)
b = tf.constant(30)
print(sess.run(a+b))

# 程序生成了一些四维数据, 然后用一个平面拟合它

# 使用 NumPy 生成假数据(phony data), 总共 1000 个点.
x_data = np.float32(np.random.rand(3, 1000)) # 随机输入  x_data是三维数据x1、x2和x3
y_data = np.dot([0.100, 0.200, 0.300], x_data) + 0.300  #y_data = 0.1*x1+0.2*x2+0.3*x3+0.3

# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
y = tf.matmul(W, x_data) + b  # tf.matmul 将矩阵 a 乘以矩阵 b,生成a * b

# 最小化方差
# 
loss = tf.reduce_mean(tf.square(y - y_data))  # reduce_mean 沿着tensor的某一维度，计算元素的平均值。由于输出tensor的维度比原tensor的低，这类操作也叫降维。
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
# 
init = tf.global_variables_initializer()
# print(sess.run(b))  # 这里只是声明了init，sess.run(init)才是真正的初始化
sess.run(init)
# print(sess.run(y))  # 这里就可以打印出来了  b,W,y都可以打印

# 拟合超平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
# 得到最佳拟合结果 W: [[ 0.09999093  0.19999063  0.29998976]], b: [ 0.30001488]
