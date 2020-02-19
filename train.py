# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import mlogger
import pickle
import os
import warnings

from dnc import dnc
from dnc import repeat_copy
from optim import get_optimizer

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_string("optimizer", "rmsprop", "Optimizer algorithm.")
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("fraction", 0.15, "l4 Optimizer fraction.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 10000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")


def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                   FLAGS.min_length, FLAGS.max_length,
                                   FLAGS.min_repeats, FLAGS.max_repeats)
  dataset_tensors = dataset()

  output_logits = run_model(dataset_tensors.observations, dataset.target_size)
  # Used for visualization.
  output = tf.round(
      tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))

  train_loss = dataset.cost(output_logits, dataset_tensors.target,
                            dataset_tensors.mask)
  train_acc = repeat_copy.masked_accuracy(output, dataset_tensors.target,
                            dataset_tensors.mask)
  target = dataset_tensors.target

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  if FLAGS.optimizer not in ('alig', 'l4mom', 'l4adam'):
    grads, _ = tf.clip_by_global_norm(
          tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)
  else:
    grads = tf.gradients(train_loss, trainable_variables)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  optimizer = get_optimizer(FLAGS)

  if FLAGS.optimizer == "alig":
    train_step = optimizer.minimize(train_loss, global_step=global_step)
    step_size = optimizer._learning_rate
  elif 'l4' in FLAGS.optimizer:
    FLAGS.learning_rate = FLAGS.fraction
    step_size = tf.constant(FLAGS.learning_rate, name='fraction')
    grads_and_vars = optimizer.compute_gradients(train_loss)
    train_step = optimizer.apply_gradients(grads_and_vars)
  else:
    step_size = step_size = tf.constant(FLAGS.learning_rate, name='step_size')
    train_step = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)

  tf.summary.scalar('loss', train_loss)
  merged_summary = tf.summary.merge_all()
  saver = tf.train.Saver()

  hooks = []
  env_name = 'dnc-{opt}-{lr}'.format(opt=FLAGS.optimizer,
                                     lr=FLAGS.learning_rate)
  if 'VISDOM_SERVER' in os.environ:
    plotter = mlogger.VisdomPlotter({'env': env_name, 'server': os.environ['VISDOM_SERVER'],
                                   'port': 9007}, manual_update=True)
  else:
    plotter = None

  xp = mlogger.Container()
  hparams = dict([(key, getattr(FLAGS, key)) for key in FLAGS.__flags])
  xp.config = mlogger.Config(plotter=plotter, **hparams)

  xp.acc = mlogger.metric.Average(plotter=plotter, plot_title='Accuracy')
  xp.loss = mlogger.metric.Average(plotter=plotter, plot_title='Loss')
  xp.log_loss = mlogger.metric.Simple(plotter=plotter, plot_title='Log-Loss')
  xp.step_size = mlogger.metric.Average(plotter=plotter, plot_title='Step-Size')

  if plotter:
    plotter.set_win_opts(title="Log-Loss", opts={'ytype': 'log'})

  # Train.
  with tf.train.SingularMonitoredSession(
      hooks=hooks) as sess:
    #   hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    start_iteration = sess.run(global_step)
    total_loss = 0
    loss_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir)

    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss, step_size_value, acc = sess.run([train_step, train_loss, step_size, train_acc])
      xp.acc.update(acc)
      xp.loss.update(loss)
      xp.step_size.update(step_size_value)
      total_loss += loss

      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np = sess.run([dataset_tensors, output])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np)
        tf.logging.info("%d: Avg training loss %f.\n%s",
                        train_iteration, total_loss / report_interval,
                        dataset_string)
        total_loss = 0
        xp.log_loss.update(xp.loss.value).log(time=train_iteration).reset()
        xp.loss.log(time=train_iteration).reset()
        xp.acc.log(time=train_iteration).reset()
        xp.step_size.log(time=train_iteration).reset()

    dirname = os.path.join('../../results/dnc', env_name)
    if os.path.exists(dirname):
      warnings.warn('An experiment already exists at {}'
                    .format(os.path.abspath(dirname)))
    else:
      os.makedirs(dirname)
    filename = os.path.join(dirname, 'results.json')
    xp.save_to(filename)


def main(unused_argv):
  tf.logging.set_verbosity(0)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()
