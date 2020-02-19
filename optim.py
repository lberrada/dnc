import tensorflow as tf

def get_optimizer(FLAGS):
  if FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)

  if FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  if FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)

  if FLAGS.optimizer == 'nag':
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)

  if FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

  if FLAGS.optimizer == 'alig':
    from alig.tf import AliG
    if FLAGS.learning_rate <= 0:
      FLAGS.learning_rate = None
    optimizer = AliG(FLAGS.learning_rate, eps=0)

  if FLAGS.optimizer == 'l4adam':
    import L4
    optimizer = L4.L4Adam(fraction=FLAGS.fraction)

  if FLAGS.optimizer == 'l4mom':
    import L4
    optimizer = L4.L4Mom(fraction=FLAGS.fraction)

  return optimizer