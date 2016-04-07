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

"""Routine for loading the (aPascal) image file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile


# Global constants describing the aPascal data set.
#tf.app.flags.DEFINE_string('apascal_root','/home/dalgu/dataset/apascal/dataset_crop_total/',
#                           """Directory where to write log and checkpoint.""")
#APASCAL_ROOT = FLAGS.apascal_root
#APASCAL_ROOT = '/data/common_datasets/apascal/dataset_crop_total/'
APASCAL_ROOT = '/home/dalgu/dataset/apascal/dataset_crop_total/'
NUM_ATTRS = 25
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = -1 # will be set after input() is called
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = -1 # will be set after input() is called

# Constants used in Caffenet
IMAGE_HEIGHT =224
IMAGE_WIDTH =224

print('[aPascal Dataset Configuration]')
print('\tDataset root: %s' % APASCAL_ROOT)
print('\tNumber of attributes: %d' % NUM_ATTRS)

def read_apascal(txt_fpath, dataset_root, shuffle=False):
  """Reads and parses examples from aPascal data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    list_fpath: Path to a txt file containing subpath of input image and labels
      line-by-line
    dataset_root: Path to the root of the dataset images.

  Returns:
    An object representing a single example, with the following fields:
      path: a scalar string Tensor of the path to the image file.
      labels: an int32 Tensor with the 64 attributes(0/1)
      image: a [height, width, depth(BGR)] float32 Tensor with the image data
  """

  class APascalRecord(object):
    pass
  result = APascalRecord()

  # Read a line from the file(list_fname)
  filename_queue = tf.train.string_input_producer([txt_fpath], shuffle=shuffle)
  text_reader = tf.TextLineReader()
  _, value = text_reader.read(filename_queue)

  # Parse the line -> filepath, labels(64)
  record_default = [['']] + [[0] for _ in range(NUM_ATTRS)]
  parsed_entries = tf.decode_csv(value, record_default, field_delim=' ')
  for i in range(1, NUM_ATTRS+1):
    parsed_entries[i] = tf.reshape(parsed_entries[i], [1])
  result.labels = tf.cast(tf.concat(0, parsed_entries[1:]), tf.int32)

  # Read image from the filepath
  # image_path = os.path.join(dataset_root, parsed_entries[0])
  dataset_root_t = tf.constant(dataset_root)
  result.image_path = dataset_root_t + parsed_entries[0] # String tensors can be concatenated by add operator
  raw_jpeg = tf.read_file(result.image_path)
  result.image = tf.image.decode_jpeg(raw_jpeg, channels=3)

  return result


def preprocess_image(input_image):
  # Preprocess the image: resize -> mean subtract -> channel swap (-> transpose X -> scale X)
  image = tf.cast(input_image, tf.float32)
  image = tf.image.resize_images(image, IMAGE_HEIGHT, IMAGE_WIDTH)
  image_R, image_G, image_B = tf.split(2, 3, image)

  blue_mean = 103.062624
  green_mean = 115.902883
  red_mean = 123.151631

  image = tf.concat(2, [image_B - blue_mean, image_G - green_mean, image_R - red_mean], name="centered_bgr")
  # image = tf.concat(2, [image_R, image_G, image_B]) # BGR -> RGB
  # imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
  # image = image - imagenet_mean # [224, 224, 3] - [3] (Subtract with broadcasting)
  # image = tf.transpose(image, [2, 0, 1]) # No transpose
  # No scaling

  return image

image_summary_added = False

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle=True):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of [NUM_ATTRS] of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Attribute labels. 2D tensor of [batch_size, NUM_ATTRS] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if not shuffle:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  else:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  #global image_summary_added
  #if not image_summary_added:
  #  tf.image_summary('images', images)
  #  image_summary_added = True

  return images, label_batch


def distorted_inputs(eval_data, batch_size, shuffle=True):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    dataset_root = os.path.join(APASCAL_ROOT, 'train', '_')[:-1]
    txt_fpath = os.path.join(APASCAL_ROOT, 'train_apy25.txt')
  else:
    dataset_root = os.path.join(APASCAL_ROOT, 'test', '_')[:-1]
    txt_fpath = os.path.join(APASCAL_ROOT, 'test_apy25.txt')

  for f in [dataset_root, txt_fpath]:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with open(txt_fpath, 'r') as fd:
    num_examples_per_epoch = len(fd.readlines())
    if not eval_data:
        global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = num_examples_per_epoch
    else:
        global NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = num_examples_per_epoch
  print('\tLoad file list from %s' % txt_fpath)
  print('\tTotal %d files' % num_examples_per_epoch)

  # Read examples from files.
  read_input = read_apascal(txt_fpath, dataset_root)

  distorted_image = tf.cast(read_input.image, tf.float32)

#  height = IMAGE_SIZE
#  width = IMAGE_SIZE
#
#  # Image processing for training the network. Note the many random
#  # distortions applied to the image.
#
#  # Randomly crop a [height, width] section of the image.
#  distorted_image = tf.image.random_crop(reshaped_image, [height, width])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

#  # Subtract off the mean and divide by the variance of the pixels.
#  float_image = tf.image.per_image_whitening(distorted_image)

  # Preprocess the image
  distorted_image = preprocess_image(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.02
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d aPascal images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(distorted_image, read_input.labels,
                                         min_queue_examples, batch_size, shuffle)


def inputs(eval_data, batch_size, shuffle=True):
  """Construct input for aPascal evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    dataset_root = os.path.join(APASCAL_ROOT, 'train', '_')[:-1]
    txt_fpath = os.path.join(APASCAL_ROOT, 'train_apy25.txt')
  else:
    dataset_root = os.path.join(APASCAL_ROOT, 'test', '_')[:-1]
    txt_fpath = os.path.join(APASCAL_ROOT, 'test_apy25.txt')

  for f in [dataset_root, txt_fpath]:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with open(txt_fpath, 'r') as fd:
    num_examples_per_epoch = len(fd.readlines())
    if not eval_data:
        global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = num_examples_per_epoch
    else:
        global NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = num_examples_per_epoch
  print('\tLoad file list from %s' % txt_fpath)
  print('\tTotal %d files' % num_examples_per_epoch)

  # Read examples from files.
  read_input = read_apascal(txt_fpath, dataset_root)

  # Preprocess the image
  image = preprocess_image(read_input.image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.02
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, read_input.labels,
                                         min_queue_examples, batch_size, shuffle)
