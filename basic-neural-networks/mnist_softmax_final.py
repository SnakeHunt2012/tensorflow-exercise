#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

from time import time
from math import sqrt
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 10

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "MNIST_data/", "Directory to put the data.")
flags.DEFINE_string("log_dir", "log/", "Directory to put the log.")
flags.DEFINE_string("checkpoint_file", "checkpoint/checkpoint", "Directory to put the checkpoint.")
flags.DEFINE_boolean("fake", False, "If true, uses fake data for unit testing.")

flags.DEFINE_integer("batch_size", 100, "Batch size. Must divide evenly into the dataset sizes.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_steps", 2000, "Number of steps to run trainer.")
flags.DEFINE_integer("hidden_one_size", 128, "Number of units in hidden layer 1.")
flags.DEFINE_integer("hidden_two_size", 32, "Number of units in hidden layer 2.")

def inference(image_tsr, hidden_one_size, hidden_two_size):

    # hidden layer one
    with tf.name_scope("hidden_one"):
        weight_var = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden_one_size],
                                                     stddev = 1.0 / sqrt(float(IMAGE_PIXELS))),
                                 name = "weight")
        bias_var = tf.Variable(tf.zeros([hidden_one_size]), name = "bias")
        hidden_one_tsr = tf.nn.relu(tf.matmul(image_tsr, weight_var) + bias_var)
    
    # hidden layer two
    with tf.name_scope("hidden_two"):
        weight_var = tf.Variable(tf.truncated_normal([hidden_one_size, hidden_two_size],
                                                     stddev = 1.0 / sqrt(float(hidden_one_size))),
                                 name = "weight")
        bias_var = tf.Variable(tf.zeros([hidden_two_size]), name = "bias")
        hidden_two_tsr = tf.nn.relu(tf.matmul(hidden_one_tsr, weight_var) + bias_var)

    # linear output layer
    with tf.name_scope("linear_output"):
        weight_var = tf.Variable(tf.truncated_normal([hidden_two_size, IMAGE_SIZE],
                                                     stddev = 1.0 / sqrt(float(hidden_two_size))),
                                 name = "weight")
        bias_var = tf.Variable(tf.zeros([IMAGE_SIZE]), name = "bias")
        linear_output_tsr = tf.matmul(hidden_two_tsr, weight_var) + bias_var
        
    return linear_output_tsr

def loss(logit_tsr, label_tsr):

    label_tsr = tf.to_int64(label_tsr)
    cross_entropy_tsr = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_tsr, label_tsr, name = "xentropy")
    loss_tsr = tf.reduce_mean(cross_entropy_tsr, name = "xentropy_mean")
    return loss_tsr

def evaluation(logit_tsr, label_tsr):

    correct_tsr = tf.nn.in_top_k(logit_tsr, label_tsr, 1)
    return tf.reduce_sum(tf.cast(correct_tsr, tf.int32))

def train(loss_tsr, learning_rate):

    tf.scalar_summary(loss_tsr.op.name, loss_tsr)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    train_op = optimizer.minimize(loss_tsr, global_step = global_step)
    return train_op

def evaluate(session, image_plr, label_plr, evaluation_tsr, data_set):

    true_count = 0.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        image_arr, label_arr = data_set.next_batch(FLAGS.batch_size, FLAGS.fake)
        feed_dict = {image_plr: image_arr, label_plr: label_arr}
        true_count += session.run(evaluation_tsr, feed_dict = feed_dict)
    accuracy = true_count / num_examples
    print "Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f" % (num_examples, true_count, accuracy)

def main():

    data_sets = input_data.read_data_sets(FLAGS.data_dir, FLAGS.fake)

    with tf.Graph().as_default():
        image_plr = tf.placeholder(tf.float32, shape = (None, IMAGE_PIXELS))
        label_plr = tf.placeholder(tf.int32, shape = (None))

        linear_output_tsr = inference(image_plr, FLAGS.hidden_one_size, FLAGS.hidden_two_size)
        loss_tsr = loss(linear_output_tsr, label_plr)
        evaluation_tsr = evaluation(linear_output_tsr, label_plr)
        
        train_op = train(loss_tsr, FLAGS.learning_rate)
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()

        saver = tf.train.Saver()
        session = tf.Session()

        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, session.graph)

        session.run(init_op)

        for step in xrange(FLAGS.max_steps):
            start_time = time()

            image_arr, label_arr = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake)
            feed_dict = {image_plr: image_arr, label_plr: label_arr}
            _, loss_arr = session.run([train_op, loss_tsr], feed_dict = feed_dict)

            duration = time() - start_time

            if step % 100 == 0:
                print "Step %d: loss = %.2f (%.3f sec)" % (step, loss_arr, duration)
                summary_str = session.run(summary_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(session, FLAGS.checkpoint_file, global_step = step)
                evaluate(session, image_plr, label_plr, evaluation_tsr, data_sets.train)
                evaluate(session, image_plr, label_plr, evaluation_tsr, data_sets.validation)
                evaluate(session, image_plr, label_plr, evaluation_tsr, data_sets.test)

if __name__ == "__main__":

    main()
