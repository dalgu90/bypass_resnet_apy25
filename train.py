# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import model

from IPython import embed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train',
                           """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 10,
                            """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 50,
                            """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 100,
                            """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95,
                            """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  print('[Training Configuration]')
  print('\tTrain dir: %s' % FLAGS.train_dir)
  print('\tTraining max steps: %d' % FLAGS.max_steps)
  print('\tSteps per displaying info: %d' % FLAGS.display)
  print('\tSteps per testing: %d' % FLAGS.test_interval)
  print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
  print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)

  """Train aPascal for a number of steps."""
  with tf.Graph().as_default():
    init_step = 0
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for aPascal.
    train_images, train_labels = model.distorted_inputs('train')
    test_images, test_labels = model.inputs('eval')

    # Build a Graph that computes the predictions from the inference model.
    images = tf.placeholder(tf.float32, [FLAGS.batch_size, model.IMAGE_WIDTH, model.IMAGE_WIDTH, 3])
    labels = tf.placeholder(tf.int32, [FLAGS.batch_size, model.NUM_ATTRS])
    probs = model.inference(images)

    # Calculate loss. (cross_entropy loss)
    loss, acc = model.loss_acc(probs, labels)
    tf.scalar_summary("accuracy", acc)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op, lr = model.train(loss, global_step)

    # Build the summary operation based on the TF collection of Summaries.
    train_summary_op = tf.merge_all_summaries()

    # Loss and accuracy summary used in test phase)
    loss_summary = tf.scalar_summary("test/loss", loss)
    acc_summary = tf.scalar_summary("test/accuracy", acc)
    test_summary_op = tf.merge_summary([loss_summary, acc_summary])

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print('\tRestore from %s' % ckpt.model_checkpoint_path)
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      print('No checkpoint file found. Start from the scratch.')
      # if finetune, load variables of the final predication layers
      # from pretrained model
      if FLAGS.finetune:
        base_variables = tf.trainable_variables()[:-2*model.NUM_ATTRS]
        base_saver = tf.train.Saver(base_variables, max_to_keep=10000)
        ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_dir)
        print('Initial checkpoint: ' + ckpt.model_checkpoint_path)
        base_saver.restore(sess, ckpt.model_checkpoint_path)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

    # Training!!
    for step in xrange(init_step, FLAGS.max_steps):
      start_time = time.time()
      try:
        train_images_val, train_labels_val = sess.run([train_images, train_labels])
        _, lr_value, loss_value, acc_value, train_summary_str = sess.run([train_op, lr, loss, acc, train_summary_op],
                                                                         feed_dict={images:train_images_val, labels:train_labels_val})
      except tf.python.framework.errors.InvalidArgumentError:
        embed()
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % FLAGS.display == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                             examples_per_sec, sec_per_batch))

        summary_writer.add_summary(train_summary_str, step)

      if step % FLAGS.test_interval == 0:
        test_images_val, test_labels_val = sess.run([test_images, test_labels])
        loss_value, acc_value, test_summary_str = sess.run([loss, acc, test_summary_op],
                                                           feed_dict={images:test_images_val, labels:test_labels_val})
        format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
        print (format_str % (datetime.now(), step, loss_value, acc_value))
        summary_writer.add_summary(test_summary_str, step)

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
