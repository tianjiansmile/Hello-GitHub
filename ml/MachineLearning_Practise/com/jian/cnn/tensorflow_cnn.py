# coding:utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
        # 使用卷积神经网络会有很多权重和偏置需要创建,我们可以定义初始化函数便于重复使用
        # 这里我们给权重制造一些随机噪声避免完全对称,使用截断的正态分布噪声,标准差为0.1
        # :param shape: 需要创建的权重Shape
        # :return: 权重Tensor
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        # 偏置生成函数,因为激活函数使用的是ReLU,我们给偏置增加一些小的正值(0.1)避免死亡节点(dead neurons)
        # :param shape:
        # :return:
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def conv2d(x, W):
        # 卷积层接下来要重复使用,tf.nn.conv2d是Tensorflow中的二维卷积函数,
        # :param x: 输入 例如[5, 5, 1, 32]代表 卷积核尺寸为5x5,1个通道,32个不同卷积核
        # :param W: 卷积的参数
        #     strides:代表卷积模板移动的步长,都是1代表不遗漏的划过图片的每一个点.
        #     padding:代表边界处理方式,SAME代表输入输出同尺寸
        # :return:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
        # tf.nn.max_pool是TensorFLow中最大池化函数.我们使用2x2最大池化
        # 因为希望整体上缩小图片尺寸,因而池化层的strides设为横竖两个方向为2步长
        # :param x:
        # :return:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def train(mnist):
        # 使用占位符  None代表未知的样本量，784代表像素28*28
        x = tf.placeholder(tf.float32, [None, 784])     # x为特征
        y_ = tf.placeholder(tf.float32, [None, 10])  # y_为label 输出为0-9的one-hot编码 输出层

        # 卷积中将1x784转换为28x28x1  [-1,,,]代表样本数量不变 [,,,1]代表通道数
        x_image = tf.reshape(x, [-1, 28, 28, 1])


        # 第一个卷积层  [5, 5, 1, 32]代表 卷积核尺寸为5x5,1个图像通道,32个不同卷积核，一个层的卷积核的通道数取决于图像通道
        # 也就是说这一次卷积操作输入是1，输出是32
        # 创建滤波器权值-->加偏置-->卷积-->池化
        W_conv1 = weight_variable([5, 5, 1, 32])
        # 偏置
        b_conv1 = bias_variable([32])

        # 先进行卷积操作，再进行线性修正，relu--- max(0,x)  得到第一卷积层
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)  # 28x28x1 与32个5x5x1滤波器 --> 28x28x32

        print(h_conv1)
        # 进行池化操作
        h_pool1 = max_pool_2x2(h_conv1) # 28x28x32 -->14x14x32

        # 第二层卷积层 卷积核依旧是5x5 通道为32   有64个不同的卷积核，第二层的卷积核通道取决于上一层卷积核的数量，上一层的卷积核数量是32，
        # 那么这一层卷积层就有32层，故本次通道是32
        # 这次卷积输入是32个，输出是64个
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #14x14x32 与64个5x5x32滤波器 --> 14x14x64
        h_pool2 = max_pool_2x2(h_conv2) #14x14x64 --> 7x7x64

        # h_pool2的大小为7x7x64 转为1-D 然后做FC层
        # 到了全连接操作，输入为64个7X7的feature矩阵，输出是1024维
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])  # 7x7x64 --> 1x3136
        # 这里的操作就不是卷积了，而是正常的线性变换 y = WX + b
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #FC层传播 3136 --> 1024

        # 使用Dropout层减轻过拟合,通过一个placeholder传入keep_prob比率控制
        # 在训练中,我们随机丢弃一部分节点的数据来减轻过拟合,预测时则保留全部数据追求最佳性能
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 将Dropout层的输出连接到一个Softmax层,得到最后的概率输出
        # 1024进 10出
        W_fc2 = weight_variable([1024, 10])  # MNIST只有10种输出可能
        b_fc2 = bias_variable([10])
        # 简单的线性操作之后，通过softmax将最后结果归一化得到当前图像的对10中可能结果的预测概率
        # softmax就相当于 [-3,2,-1,0]-->[0.0057,0.8390,0.0418,0.1135] 给出一个归一化之后的概率值
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 1024 --> 10

        # 定义损失函数,依旧使用交叉熵  同时定义优化器  learning rate = 1e-4
        # 交叉熵刻画了两个概率分布之间的距离，是分类问题中使用比较广泛的损失函数
        # 假设有一个三分类问题，某个样例的正确答案是（1，0，0）。某模型经过softmax回归之后的预测答案是(0.5,0.4,0.1)，
        # 那么这个预测和正确答案之间的交叉熵为：H（（1，0，0），（0.5，0.4，0.1）)=-(1*log0.5+0*log0.4+0*log0.1)≈0.3
        #                                  H（（1，0，0），（0.8，0.1，0.1）)=-（1*log0.8）≈0.1
        # 结果约接近正确答案，交叉熵越小，其实看log函数就能看出端倪，正确结果_y与预测结果y_conv的乘积就是 loss = -1xlog(p) p越接近与q loss越接近0
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # 定义评测准确率
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #开始训练
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            init_op = tf.global_variables_initializer() #初始化所有变量
            sess.run(init_op)

            STEPS = 200
            for i in range(STEPS):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d,training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print("test accuracy %g" % acc)



if __name__=="__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 载入数据集
    train(mnist)