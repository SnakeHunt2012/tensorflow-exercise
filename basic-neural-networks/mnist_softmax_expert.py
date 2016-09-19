#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def main():
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as session:
        with tf.device("/gpu:1"):
            X_tsr = tf.placeholder(tf.float32, shape = [None, 784])
            X_image_tsr = tf.reshape(X_tsr, [-1, 28, 28, 1])
            
            W_conv1_var = weight_variable([5, 5, 1, 32])
            b_conv1_var = bias_variable([32])
            
            h_conv1_tsr = tf.nn.relu(conv2d(X_image_tsr, W_conv1_var) + b_conv1_var)
            h_pool1_tsr = max_pool_2x2(h_conv1_tsr)
            
        with tf.device("/gpu:2"):
            W_conv2_var = weight_variable([5, 5, 32, 64])
            b_conv2_var = bias_variable([64])
            
            h_conv2_tsr = tf.nn.relu(conv2d(h_pool1_tsr, W_conv2_var) + b_conv2_var)
            h_pool2_tsr = max_pool_2x2(h_conv2_tsr)
            h_pool2_flat_tsr = tf.reshape(h_pool2_tsr, [-1, 7 * 7 * 64])
            
        with tf.device("/gpu:3"):
            W_fc1_var = weight_variable([7 * 7 * 64, 1024])
            b_fc1_var = bias_variable([1024])
            
            h_fc1_tsr = tf.nn.relu(tf.matmul(h_pool2_flat_tsr, W_fc1_var) + b_fc1_var)
            
            keep_proba_tsr = tf.placeholder(tf.float32)
            h_fc1_drop_tsr = tf.nn.dropout(h_fc1_tsr, keep_proba_tsr)
            
            W_fc2_var = weight_variable([1024, 10])
            b_fc2_var = bias_variable([10])
            
            y_true_tsr = tf.placeholder(tf.float32, shape = [None, 10])
            y_pred_tsr = tf.nn.softmax(tf.matmul(h_fc1_drop_tsr, W_fc2_var) + b_fc2_var)
            
            cross_entropy_tsr = tf.reduce_mean(-tf.reduce_sum(y_true_tsr * tf.log(y_pred_tsr), reduction_indices = [1]))
            correct_tsr = tf.equal(tf.argmax(y_pred_tsr, 1), tf.argmax(y_true_tsr, 1))
            accuracy_tsr = tf.reduce_mean(tf.cast(correct_tsr, tf.float32))
             
        train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_tsr)
        session.run(tf.initialize_all_variables())
        for i in xrange(20000):
            X_arr, y_arr = mnist.train.next_batch(50)
            if i % 100 == 0:
                accuracy_arr = accuracy_tsr.eval(feed_dict = {X_tsr: X_arr, y_true_tsr: y_arr, keep_proba_tsr: 1.0})
                print "step: %d, training accuracy %g" % (i, accuracy_arr)
            train_op.run(feed_dict = {X_tsr: X_arr, y_true_tsr: y_arr, keep_proba_tsr: 0.5})
        
        print "testing accuracy %g" % accuracy_tsr.eval(feed_dict = {X_tsr: mnist.test.images, y_true_tsr: mnist.test.labels, keep_proba_tsr: 1.0})
            

if __name__ == "__main__":

    main()
