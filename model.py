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

"""Builds the Caffenet network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import resnet.resnet as resnet
import resnet.utils as utils

from IPython import embed

import cPickle as pickle
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('l2_weight', 0.0001,
                            """L2 loss weight applied to all the weights
                          except the last fc layers""")
tf.app.flags.DEFINE_float('l1_weight', 0.001,
                            """L1 loss weight applied to the last fc layers""")
tf.app.flags.DEFINE_float('initial_lr', 0.1,
                            """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 50.0,
                            """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1,
                            """Learning rate decay factor""")
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0,
                            """Basenet gradient will be clipped to maximally this norm""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                            """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_boolean('basenet_train', True,
                            """Flag whether the model will train the base network""")
tf.app.flags.DEFINE_float('basenet_lr_ratio', 0.1,
                            """Learning rate ratio of basenet to bypass net""")
tf.app.flags.DEFINE_boolean('finetune', False,
                            """Flag whether the L1 connection weights will be only made at
                            the position where the original bypass network has nonzero
                            L1 connection weights""")
tf.app.flags.DEFINE_string('pretrained_dir', './pretrain',
                           """Directory where to load pretrained model.(Only for
                           --finetune True""")

import awa_input as data_input


print('[Network Configuration]')
# Batch size
print('\tBatch size: %d' % FLAGS.batch_size)

# Global constants describing the CIFAR-10 data set.
IMAGE_HEIGHT = data_input.IMAGE_HEIGHT
IMAGE_WIDTH = data_input.IMAGE_WIDTH
NUM_ATTRS = data_input.NUM_ATTRS

# Constants describing the network
L2_LOSS_WEIGHT = FLAGS.l2_weight
L1_LOSS_WEIGHT = FLAGS.l1_weight
print('\tL2 loss weight: %f' % L2_LOSS_WEIGHT)
print('\tL1 loss weight: %f' % L1_LOSS_WEIGHT)
BASENET_TRAIN = FLAGS.basenet_train
print('\tBasenet Training: %s' % str(BASENET_TRAIN))
BASENET_LR_RATIO = FLAGS.basenet_lr_ratio
if BASENET_TRAIN:
  print('\tBasenet lr ratio: %f' % BASENET_LR_RATIO)

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.99     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = FLAGS.lr_step_epoch      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = FLAGS.lr_decay  # Learning rate decay factor.
MOMENTUM = FLAGS.momentum  # Momentum.
INITIAL_LEARNING_RATE = FLAGS.initial_lr       # Initial learning rate.
MAX_GRADIENT_NORM = FLAGS.max_gradient_norm
print('\tMoving average decay: %f' % MOVING_AVERAGE_DECAY)
print('\tNumber of epochs per decay: %f' % NUM_EPOCHS_PER_DECAY)
print('\tLearning rate decay factor: %f' % LEARNING_RATE_DECAY_FACTOR)
print('\tInitial learning rate %f' % INITIAL_LEARNING_RATE)
print('\tMax gradient norm %f' % MAX_GRADIENT_NORM)

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
def _histogram_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)

def _sparsity_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _almost_sparsity_summary(x, eps):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    eps_t = tf.constant(eps, dtype=x.dtype, name="threshold")
    fraction = tf.reduce_mean(tf.cast(tf.less(tf.abs(x), eps_t), tf.float32))
    tf.scalar_summary(tensor_name + '/less_than_' + ("%g" % eps), fraction)


def distorted_inputs(data_class, shuffle=True):
    return data_input.distorted_inputs(data_class=data_class,
                                       batch_size=FLAGS.batch_size,
                                       shuffle=shuffle)


def inputs(data_class, shuffle=True):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
      shuffle: bool, to shuffle dataset list to read

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    return data_input.inputs(data_class=data_class,
                             batch_size=FLAGS.batch_size,
                             shuffle=shuffle)


def inference(images):
  """Build the CaffeNet model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  ###### Pretrained ResNet ####
  resnet_model = resnet.Model(50, BASENET_TRAIN)
  resnet_model.build(images)

  graph = tf.get_default_graph()
  conv1 = graph.get_operation_by_name("conv1/relu").outputs[0]
  res2a = graph.get_operation_by_name("res2a/relu").outputs[0]
  res2b = graph.get_operation_by_name("res2b/relu").outputs[0]
  res2c = graph.get_operation_by_name("res2c/relu").outputs[0]
  res3a = graph.get_operation_by_name("res3a/relu").outputs[0]
  res3b = graph.get_operation_by_name("res3b/relu").outputs[0]
  res3c = graph.get_operation_by_name("res3c/relu").outputs[0]
  res3d = graph.get_operation_by_name("res3d/relu").outputs[0]
  res4a = graph.get_operation_by_name("res4a/relu").outputs[0]
  res4b = graph.get_operation_by_name("res4b/relu").outputs[0]
  res4c = graph.get_operation_by_name("res4c/relu").outputs[0]
  res4d = graph.get_operation_by_name("res4d/relu").outputs[0]
  res4e = graph.get_operation_by_name("res4e/relu").outputs[0]
  res4f = graph.get_operation_by_name("res4f/relu").outputs[0]
  res5a = graph.get_operation_by_name("res5a/relu").outputs[0]
  res5b = graph.get_operation_by_name("res5b/relu").outputs[0]
#  res5c = graph.get_operation_by_name("res5c/relu").outputs[0] # Not used due to the duplication of pool5
  pool5 = graph.get_operation_by_name("pool5").outputs[0]
  res_prob = graph.get_operation_by_name("prob").outputs[0]

  ###### Bypass layers
  # Attach fc layers to all (pooled) layers
  base_res_layers = [conv1, res2a, res2b, res2c, res3a, res3b, res3c, res3d, res4a,
                     res4b, res4c, res4d, res4e, res4f, res5a, res5b]
  base_res_layer_names = ["conv1", "res2a", "res2b", "res2c", "res3a", "res3b", "res3c", "res3d", "res4a",
                          "res4b", "res4c", "res4d", "res4e", "res4f", "res5a", "res5b"]
  bypass_pool_sizes = [14] + [17] * 3 + [12] * 4 + [8] * 6 + [7] * 2
  bypass_pool_strides = [14] + [13] * 3 + [8] * 4 + [6] * 6 + [1] * 2
  bypass_layers = []
  with tf.variable_scope('bypass') as scope:
    for layer, name, pool_size, pool_stride in zip(base_res_layers, base_res_layer_names, bypass_pool_sizes, bypass_pool_strides):
      with tf.variable_scope(name):
        # pool = tf.nn.max_pool(layer, ksize=[1, layer.get_shape()[1], layer.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        pool = tf.nn.avg_pool(layer, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='VALID')
        flattened = tf.reshape(pool, [pool.get_shape()[0].value, pool.get_shape()[1].value * pool.get_shape()[2].value * pool.get_shape()[3].value])
        weights = utils.tf_variable_weight_decay('weights', [flattened.get_shape()[1].value, data_input.NUM_ATTRS], tf.truncated_normal_initializer(stddev=0.01))
        biases = utils.tf_variable('biases', [data_input.NUM_ATTRS], tf.constant_initializer(value=0.0))
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, weights), biases), name="fc")
#        _histogram_summary(weights)
        bypass_layers.append(fc)
    with tf.variable_scope('pool5') as scope:
      flattened = tf.reshape(pool5, [pool5.get_shape()[0].value, pool5.get_shape()[3].value])
      weights = utils.tf_variable_weight_decay('weights', [flattened.get_shape()[1].value, data_input.NUM_ATTRS], tf.truncated_normal_initializer(stddev=0.01))
      biases = utils.tf_variable('biases', [data_input.NUM_ATTRS], tf.constant_initializer(value=0.0))
      fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, weights), biases), name="fc")
#      _histogram_summary(weights)
      bypass_layers.append(fc)
    with tf.variable_scope('res_prob') as scope:
      weights = utils.tf_variable_weight_decay('weights', [res_prob.get_shape()[1].value, data_input.NUM_ATTRS], tf.truncated_normal_initializer(stddev=0.01))
      biases = utils.tf_variable('biases', [data_input.NUM_ATTRS], tf.constant_initializer(value=0.0))
      fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(res_prob, weights), biases), name="fc")
#      _histogram_summary(weights)
      bypass_layers.append(fc)

  print('\tTotal %d bypass layers' % len(bypass_layers))

  # Split along axis-1(attribute-wise)
  fc_slice_ll = []
  for fc in bypass_layers:
    fc_slice_ll.append(tf.split(1, data_input.NUM_ATTRS, fc))

  ##### Selection layers
  prob_list = []
  lasso_weights_list = []
  # 1. Pretrain - Each classes/attrs are connected to all bypass layers
  if not FLAGS.finetune:
    # Concatenate and output(sigmoid)
    # Check the weight sparsity
    with tf.variable_scope('prob') as scope:
      for i in range(data_input.NUM_ATTRS):
        concat = tf.concat(1, [slices[i] for slices in fc_slice_ll], name=("concat%d"%(i+1)) )
        weights = utils.tf_variable_lasso(('weights%d'%(i+1)), [len(bypass_layers), 1], tf.truncated_normal_initializer(stddev= 0.01))
        biases = utils.tf_variable(('biases%d'%(i+1)), [1], tf.constant_initializer())
        prob_list.append(tf.sigmoid(tf.nn.bias_add(tf.matmul(concat, weights), biases), name=("prob%d"%(i+1))))
        lasso_weights_list.append(weights)
  # 2. Finetune - Each classes/attrs are connected to layers only with nonzero
  # pretrain weights
  else:
    # Concatenate and output(sigmoid)
    # Check the weight sparsity
    with open(os.path.join(FLAGS.pretrained_dir, 'l1_weight.pkl'), 'r') as fd:
      train_weight_dict = pickle.load(fd)
    with open(data_input.ATTRIBUTE_LIST_FPATH, 'r') as fd:
      predicate_list = [temp.strip().split()[1] for temp in fd.readlines()]
    with tf.variable_scope('prob') as scope:
      for i in range(data_input.NUM_ATTRS):
        # only take slices of predictions with non-zero training weights
        train_weight = train_weight_dict[predicate_list[i]][0]
        if(np.sum(np.abs(train_weight)) > 0): # not all weights are zero -> connect layers of non-zero weight(but the weights are initialized)
          nonzero_slices = [fc_slice_ll[j][i] for j in range(len(bypass_layers)) if train_weight[j]]
          concat = tf.concat(1, nonzero_slices)
          weights = utils.tf_variable_weight_decay(('weights%d'%(i+1)), [len(nonzero_slices), 1], tf.truncated_normal_initializer(stddev= 0.01))
        else: # all weights are zero! -> connect all layers with L1 lasso loss
          concat = tf.concat(1, [slices[i] for slices in fc_slice_ll], name=("concat%d"%(i+1)) )
          weights = utils.tf_variable_lasso(('weights%d'%(i+1)), [len(bypass_layers), 1], tf.truncated_normal_initializer(stddev= 0.01))
          lasso_weights_list.append(weights)
        biases = utils.tf_variable(('biases%d'%(i+1)), [1], tf.constant_initializer())
        prob_list.append(tf.sigmoid(tf.nn.bias_add(tf.matmul(concat, weights), biases), name=("prob%d"%(i+1))))

  # Concatenate all the predictions along axis-1
  prob = tf.concat(1, prob_list, name=scope.name)
  # Concatenate all L1 weights and make histogram
  if(len(lasso_weights_list) > 0):
    lasso_weights_concat = tf.concat(1, lasso_weights_list, name='l1_weights')
    _histogram_summary(lasso_weights_concat)
    _sparsity_summary(lasso_weights_concat)
    _almost_sparsity_summary(lasso_weights_concat, 10**-7)
    _almost_sparsity_summary(lasso_weights_concat, 10**-5)

  return prob

ce_loss_flag = False
def loss_acc(probs, labels):
  """
  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate pos/neg loss weights for each attributes
  with open(data_input.TRAIN_POSNEG_FPATH, 'r') as fd:
    posneg_ll = [map(int, line.strip().split()) for line in fd.readlines()]
  #loss_posneg = [[np.sqrt(n), np.sqrt(p)]/(np.sqrt(p)+np.sqrt(n)) for p, n in posneg_ll]
  loss_posneg = [[np.float64(p+n)/np.sqrt(p), np.float64(p+n)/np.sqrt(n)]/(np.sqrt(p)+np.sqrt(n)) for p, n in posneg_ll]
  pos_weights = np.array([temp[0] for temp in loss_posneg])
  neg_weights = np.array([temp[1] for temp in loss_posneg])
  pos_weights_t = tf.constant(pos_weights, dtype=tf.float32)
  neg_weights_t = tf.constant(neg_weights, dtype=tf.float32)

  # Calculate cross-entropy loss for each prob(of attribute)
  labels_float = tf.cast(labels, tf.float32)
  probs_clip = tf.clip_by_value(probs, 0.000001, 0.999999)
  cross_entropy = -labels_float * tf.log(probs_clip) * pos_weights_t - (1.0 - labels_float) * tf.log(1.0 - probs_clip) * neg_weights_t

  # Average cross-entropy loss over the attrs. and the batch
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  print('\tCross-entropy loss added: %s' % cross_entropy_mean.name)

  # Calculate accuracy
  output_shape = [FLAGS.batch_size, data_input.NUM_ATTRS]
  zeros = tf.constant(np.zeros(output_shape), dtype=tf.float32)
  ones = tf.constant(np.ones(output_shape), dtype=tf.float32)

  preds = tf.select(probs < 0.5, zeros, ones)
  correct = tf.select(tf.equal(labels_float, preds), ones, zeros)
  accuracy = tf.reduce_mean(correct)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss) plus lasso loss term (L1 loss).
  return cross_entropy_mean, accuracy


def _add_loss_summaries(total_loss):
  """Add summaries for losses in Caffenet model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply([total_loss])

  tf.scalar_summary(total_loss.op.name + ' (raw)', total_loss)
  tf.scalar_summary(total_loss.op.name, loss_averages.average(total_loss))

#  # Attach a scalar summary to all individual losses and the total loss; do the
#  # same for the averaged version of the losses.
#  for l in losses + [total_loss]:
#    # Name each loss as '(raw)' and name the moving average version of the loss
#    # as the original loss name.
#    tf.scalar_summary(l.op.name +' (raw)', l)
#    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train caffenet model for aPascal.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  BASENET_TR_VAR_NUM = 161

  # Compute gradients.
  all_tr_vars = tf.trainable_variables()
  with tf.control_dependencies([loss_averages_op]):
    grads = []
    if BASENET_TRAIN:
      basenet_tr_vars = all_tr_vars[:BASENET_TR_VAR_NUM]
      basenet_opt = tf.train.MomentumOptimizer(lr * BASENET_LR_RATIO, MOMENTUM)
      basenet_grads = basenet_opt.compute_gradients(total_loss, basenet_tr_vars)
#      basenet_g_list, basenet_v_list = zip(*basenet_grads)
#      basenet_clip_g_list, _ = tf.clip_by_global_norm(basenet_g_list, MAX_GRADIENT_NORM)
#      basenet_grad = zip(basenet_clip_g_list, basenet_v_list)

#      for g, v in basenet_grads:
#        _histogram_summary(g) # gradient check
#        _histogram_summary(v) # variable check

      grads = grads + basenet_grads

    if BASENET_TRAIN:
      bypass_tr_vars = all_tr_vars[BASENET_TR_VAR_NUM:]
    else:
      bypass_tr_vars = all_tr_vars
    bypass_opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
    bypass_grads = bypass_opt.compute_gradients(total_loss, bypass_tr_vars)
#    bypass_g_list, bypass_v_list = zip(*bypass_grads)
#    bypass_clip_g_list, _ = tf.clip_by_global_norm(bypass_g_list, MAX_GRADIENT_NORM)
#    bypass_grad = zip(bypass_clip_g_list, bypass_v_list)

#    for g, v in bypass_grads:
#        if v.name.startswith("prob/weights"):
#            _histogram_summary(g)
#            _histogram_summary(v)

    grads = grads + bypass_grads

  # Check numerics of tensors
  # verify_tensor_list = []
  # for i in xrange(len(grads), 0, -1):
    # v, g = grads[i-1]
    # verify_tensor_list.append(tf.check_numerics(v, v.name))
    # verify_tensor_list.append(tf.check_numerics(g, g.name))

  # Apply gradients.
  apply_grad_op_list = []
  # with tf.control_dependencies(verify_tensor_list):
  if BASENET_TRAIN:
    basenet_apply_grad_op = basenet_opt.apply_gradients(basenet_grads,
                                                        global_step=global_step)
    apply_grad_op_list.append(basenet_apply_grad_op)
  bypass_apply_grad_op = bypass_opt.apply_gradients(bypass_grads,
                                                    global_step=global_step)
  apply_grad_op_list.append(bypass_apply_grad_op)

  # If basenet weights are trained together keep the basenet weights
  resnet_vars = []
  if BASENET_TRAIN and L2_LOSS_WEIGHT > 0:
    resnet_vars = all_tr_vars[:BASENET_TR_VAR_NUM]

  # Apply weight decay for the variables with l2 loss
  # If basenet weights are trained together, do not set a weight decay on the
  # conv layers of the basenet
  l2_op_list = []
  l1_op_list = []
  with tf.control_dependencies(apply_grad_op_list):
    if L2_LOSS_WEIGHT > 0:
      for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
        if var in resnet_vars[:-2]: # exclude the weights, bias of the last fc layer
          continue
          #l2_weight = L2_LOSS_WEIGHT * BASENET_LR_RATIO
        else:
          l2_weight = L2_LOSS_WEIGHT
        assign_op = var.assign_add(- lr * tf.convert_to_tensor(l2_weight) * var)
        l2_op_list.append(assign_op)
        print('\tL2 loss added: %s(strength: %f)' % (var.name, l2_weight))

    # Apply proximal gradient for the variables with l1 lasso loss
    # Non-negative weights constraint
    if L1_LOSS_WEIGHT > 0:
      for var in tf.get_collection(utils.LASSO_KEY):
        th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(L1_LOSS_WEIGHT) * lr)
        zero_t = tf.zeros(tf.shape(var))
        var_temp = var - th_t * tf.sign(var)
        assign_op = var.assign(tf.select(tf.less(var, th_t), zero_t, var_temp))
        l1_op_list.append(assign_op)
        print('\tL1 loss added: %s(strength: %f)' % (var.name, L1_LOSS_WEIGHT))


  # Add histograms for trainable variables.
#  for var in tf.trainable_variables():
#    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
#  for grad, var in grads:
#    if grad:
#      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
#  variable_averages = tf.train.ExponentialMovingAverage(
#      MOVING_AVERAGE_DECAY, global_step)
#  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies(l2_op_list + l1_op_list):
    train_op = tf.no_op(name='train')

  return train_op, lr
