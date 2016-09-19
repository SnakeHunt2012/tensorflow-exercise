#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    x_tsr = tf.placeholder(tf.float32, [None, 784])
    
    W_var = tf.Variable(tf.zeros([784, 10]))
    b_var = tf.Variable(tf.zeros([10]))
    
    y_pred_tsr = tf.nn.softmax(tf.matmul(x_tsr, W_var) + b_var)
    y_true_tsr = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy_tsr = tf.reduce_mean(-tf.reduce_sum(y_true_tsr * tf.log(y_pred_tsr), reduction_indices = [1]))
    
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_tsr)
    
    init_op = tf.initialize_all_variables()
    
    session = tf.Session()
    session.run(init_op)
    
    for i in range(1000):
        x_arr, y_arr = mnist.train.next_batch(100)
        session.run(train_op, feed_dict = {x_tsr: x_arr, y_true_tsr: y_arr})
    
    correct_tsr = tf.equal(tf.argmax(y_pred_tsr, 1), tf.argmax(y_true_tsr, 1))
    accuracy_tsr = tf.reduce_mean(tf.cast(correct_tsr, tf.float32))
    print session.run(accuracy_tsr, feed_dict = {x_tsr: mnist.test.images, y_true_tsr: mnist.test.labels})

if __name__ == "__main__":
    
    main()
