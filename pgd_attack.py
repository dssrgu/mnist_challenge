"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start

        self.x_nat_input = tf.placeholder(dtype=tf.float32,
                                          shape=[None, 784])
        self.x_adv_input = tf.placeholder(dtype=tf.float32,
                                          shape=[None, 784])
        self.y_input = tf.placeholder(dtype=tf.int64,
                                      shape=[None])

        nat_logits = self.model(self.x_nat_input)
        adv_logits = self.model(self.x_adv_input)

        _, _, xent = self.model.get_loss_values(adv_logits, self.y_input)

        self.loss = xent

        self.grad = tf.gradients(self.loss, self.x_adv_input)[0]

        # trades function
        _, y_kl, _, _ = self.model.get_trades_values(adv_logits, nat_logits)

        self.trades_grad = tf.gradients(y_kl, self.x_adv_input)[0]

    def perturb(self, x_nat, y, sess, trades=False):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if trades:
            x = x_nat + np.random.randn(*x_nat.shape) * 0.001
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1)  # ensure valid pixel range
        else:
            if self.rand:
                x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            else:
                x = np.copy(x_nat)

        for i in range(self.k):
            if trades:
                grad = sess.run(self.trades_grad, feed_dict={self.x_adv_input: x,
                                                             self.x_nat_input: x_nat})
            else:
                loss, grad = sess.run([self.loss, self.grad], feed_dict={self.x_adv_input: x,
                                                                         self.y_input: y})

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1)  # ensure valid pixel range

        loss, grad = sess.run([self.loss, self.grad], feed_dict={self.x_adv_input: x,
                                                                 self.y_input: y})
        return x


if __name__ == '__main__':
    import json
    import sys
    import math

    from tensorflow.examples.tutorials.mnist import input_data

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model()
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
    saver = tf.train.Saver()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = []  # adv accumulator

        print('Iterating over {} batches'.format(num_batches))

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = mnist.test.images[bstart:bend, :]
            y_batch = mnist.test.labels[bstart:bend]

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
