# 原文地址 https://www.cnblogs.com/welhzh/p/6607581.html
import tensorflow as tf

# tf.nn.conv2d是TensorFlow里面实现卷积的函数
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
# 具有[batch, in_height, in_width, in_channels]这样的shape，
# 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
# 注意这是一个4维的Tensor，要求类型为float32和float64其中之一
# 
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
# 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
# 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
# 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
# 
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
# 
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
# 
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true,
# 结果返回一个Tensor，这个输出，就是我们常说的feature map
# 

#case 2
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 3
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 4
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
#case 5
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
#case 6
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
#case 7
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
#case 8
input = tf.Variable(tf.random_normal([10,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("case 2")
    print(sess.run(input))
    print(sess.run(op2).shape)
    print("case 3")
    print(sess.run(op3).shape)
    print("case 4")
    print(sess.run(op4).shape)
    print("case 5")
    print(sess.run(op5).shape)
    print("case 6")
    print(sess.run(op6).shape)
    print("case 7")
    print(sess.run(op7).shape)
    print("case 8")
    print(sess.run(op8).shape)
