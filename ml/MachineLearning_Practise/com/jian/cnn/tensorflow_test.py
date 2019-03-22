# coding:utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#
def first_test():
    # 1. 定义数据 用Tensorflow计算a=(b+c)∗(c+2)
    # 首先，创建一个TensorFlow常量=>2
    const = tf.constant(2.0, name='const')

    # 创建TensorFlow变量b和c
    # 如上，TensorFlow中，使用tf.constant()定义常量，使用tf.Variable()定义变量。Tensorflow可以自动进行数据类型检测，比如：赋值2.0就默认为tf.float32，但最好还是显式地定义
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, dtype=tf.float32, name='c')

    # 定义运算(也称TensorFlow operation)
    # 创建operation
    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    # ！！TensorFlow中所有的变量必须经过初始化才能使用，初始化方式分两步：
    # 定义初始化operation
    # 运行初始化operation

    # 1. 定义init operation
    init_op = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:
        # 2. 运行init operation
        sess.run(init_op)
        # 计算
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))


if __name__ == '__main__':
    first_test()