"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model configuration
    parser.add_argument('--model_dir', default='madry', type=str)
    parser.add_argument('--data_path', default='/data/home/gaon/lazy-attack/cifar10_data', type=str)
    # training setting
    parser.add_argument('--random_seed', default=4557077)
    parser.add_argument('--max_num_training_steps', default=80000, type=int)
    parser.add_argument('--num_output_steps', default=100, type=int)
    parser.add_argument('--num_summary_steps', default=100, type=int)
    parser.add_argument('--num_checkpoint_steps', default=100, type=int)
    parser.add_argument('--training_batch_size', default=128, type=int)
    # args.trades setting
    parser.add_argument('--trades', action='store_true', help='use args.trades method')
    parser.add_argument('--beta', default=6.0, type=float, help='lambda for args.trades')
    # grad_reg setting
    parser.add_argument('--grad_reg', action='store_true', help='use gradient regularization method')
    parser.add_argument('--alpha', default=6.0, type=float, help='lambda for grad_reg')
    # attack setting
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--num_steps', default=40, type=int)
    parser.add_argument('--step_size', default=0.01, type=float)
    parser.add_argument('--loss_func', default='xent', type=str)
    parser.add_argument('--store_adv_path', default='attack.npy', type=str)
    args = parser.parse_args()

    for key, val in vars(args).items():
        print('{} = {}'.format(key, val))

    assert not (args.trades and args.grad_reg)

# Setting up training parameters
tf.set_random_seed(args.random_seed)

max_num_training_steps = args.max_num_training_steps
num_output_steps = args.num_output_steps
num_summary_steps = args.num_summary_steps
num_checkpoint_steps = args.num_checkpoint_steps
batch_size = args.training_batch_size
beta = args.beta
alpha = args.alpha

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()

model = Model()

x_nat_input = tf.placeholder(dtype=tf.float32,
                             shape=[None, 784])
x_adv_input = tf.placeholder(dtype=tf.float32,
                             shape=[None, 784])
y_input = tf.placeholder(dtype=tf.int64,
                         shape=[None])

nat_logits = model(x_nat_input)
adv_logits = model(x_adv_input)

_, _, _, nat_acc = model.get_pred_values(nat_logits, y_input)
_, _, _, adv_acc = model.get_pred_values(adv_logits, y_input)
_, _, nat_mean_xent = model.get_loss_values(nat_logits, y_input)
_, _, adv_mean_xent = model.get_loss_values(adv_logits, y_input)
_, _, _, mean_kl = model.get_trades_values(adv_logits, nat_logits)
grad_reg_loss = model.get_grad_reg_values(x_adv_input, y_input)

if args.trades:
    total_loss = nat_mean_xent + mean_kl * beta
elif args.grad_reg:
    total_loss = nat_mean_xent + grad_reg_loss * alpha
else:
    total_loss = adv_mean_xent

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       args.epsilon,
                       args.num_steps,
                       args.step_size,
                       True,
                       args.loss_func)

# Setting up the Tensorboard and checkpoint outputs
model_dir = 'models/' + args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', adv_acc)
tf.summary.scalar('accuracy adv', adv_acc)
tf.summary.scalar('xent adv train', adv_mean_xent / batch_size)
tf.summary.scalar('xent adv', adv_mean_xent / batch_size)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess, trades=args.trades)
        end = timer()
        training_time += end - start

        full_dict = {x_nat_input: x_batch,
                     x_adv_input: x_batch_adv,
                     y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc_batch, adv_acc_batch, xent_batch, kl_batch, grad_reg_loss_batch = sess.run(
                [nat_acc, adv_acc, adv_mean_xent if not (args.trades or args.grad_reg) else nat_mean_xent,
                 mean_kl, grad_reg_loss], feed_dict=full_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc_batch * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc_batch * 100))
            print('    training xent loss {:.4}'.format(xent_batch))
            print('    training kl loss {:.4}'.format(kl_batch))
            print('    training grad_reg loss {:.4}'.format(grad_reg_loss_batch))
            if args.trades:
                x_batch_pgd_adv = attack.perturb(x_batch, y_batch, sess, trades=False)
                adv_acc_batch = sess.run(adv_acc, feed_dict={x_adv_input: x_batch_pgd_adv,
                                                             y_input: y_batch})
                print('    training robust accuracy {:.4}%'.format(adv_acc_batch * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=full_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=full_dict)
        end = timer()
        training_time += end - start
