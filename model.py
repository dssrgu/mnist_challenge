"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):

    def __init__(self):

        # first convolutional layer
        self.W_conv1 = self._weight_variable([5,5,1,32])
        self.b_conv1 = self._bias_variable([32])

        # second convolutional layer
        self.W_conv2 = self._weight_variable([5,5,32,64])
        self.b_conv2 = self._bias_variable([64])

        # first fully connected layer
        self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self._bias_variable([1024])

        # output layer
        self.W_fc2 = self._weight_variable([1024,10])
        self.b_fc2 = self._bias_variable([10])

    def __call__(self, x_input):

        self.x_image = tf.reshape(x_input, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(self._conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.pre_softmax = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

        return self.pre_softmax

    def get_pred_values(self, logits, y_input):

        with tf.variable_scope('preds', reuse=tf.AUTO_REUSE):

            predictions = tf.argmax(logits, 1)
            correct_prediction = tf.equal(predictions, y_input)
            num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return predictions, correct_prediction, num_correct, accuracy

    def get_loss_values(self, logits, y_input):

        with tf.variable_scope('costs', reuse=tf.AUTO_REUSE):

            y_oh = self._one_hot(y_input)
            y_xent = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_oh, logits=logits)
            xent = tf.reduce_sum(y_xent)
            mean_xent = tf.reduce_mean(y_xent)

        return y_xent, xent, mean_xent

    def get_trades_values(self, input_logits, target_logits):

        with tf.variable_scope('trades', reuse=tf.AUTO_REUSE):
            log_softmax = tf.nn.log_softmax(input_logits)
            target_softmax = tf.nn.softmax(target_logits)
            y_kl = self._kl_divergence(log_softmax, target_softmax)
            kl = tf.reduce_sum(y_kl)
            mean_kl = tf.reduce_mean(y_kl)

        return log_softmax, y_kl, kl, mean_kl

    def get_grad_reg_values(self, x, y):

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            logits = self(x)
        with tf.variable_scope('grad_reg', reuse=tf.AUTO_REUSE):
            y_oh = self._one_hot(y)
            y_xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_oh)
            #xent = tf.reduce_sum(y_xent, name='y_xent')
            mean_xent = tf.reduce_mean(y_xent)
            grad_loss = tf.nn.l2_loss(tf.gradients(mean_xent, x)[0])

        return grad_loss

    @staticmethod
    def _kl_divergence(inputs, targets):
        # following pyTorch ver. implementation
        # inputs: log-probability
        # targets: probability
        return targets * (tf.log(targets)-inputs)

    @staticmethod
    def _one_hot(y, num_classes=10):
        one_hot = tf.one_hot(y, num_classes)
        return one_hot

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME')
