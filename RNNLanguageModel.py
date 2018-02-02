#
# RNN model for sequence learning
#
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from datetime import datetime
from tqdm import trange, tqdm
tqdm.monitor_interval = 0

class RNNLanguageModel(object):

  def __init__(self,
               vocab_size = 10,
               num_units = 2048,
               num_layers = 3,
               step_size = 32,
               batch_size = 192,
               log_dir = '/tmp/RNNLanguageModel/prueba'):
    
    self.vocab_size = vocab_size
    self.num_units = num_units
    self.num_layers = num_layers
    self.step_size = step_size
    self.log_dir = log_dir
    self.batch_size = batch_size

    tf.reset_default_graph()
    tf.set_random_seed(0)

    #
    # model graph
    #
    with tf.name_scope("input_layer"):
      X = tf.placeholder(tf.uint8, [batch_size, step_size], name='X')  # [ batch_size, step_size ]
      Y = tf.placeholder(tf.uint8, [batch_size, step_size], name='Y')  # [ batch_size, step_size ]
      X_one_hot = tf.one_hot(X, vocab_size, 1.0, 0.0)       # [ batch_size, step_size, vocab_size ]
      Y_one_hot = tf.one_hot(Y, vocab_size, 1.0, 0.0)       # [ batch_size, step_size, vocab_size ]

    with tf.name_scope("hidden_layers"):
      multicell = rnn.MultiRNNCell([rnn.GRUCell(num_units) for _ in range(num_layers)])
      initial_state = multicell.zero_state(batch_size, dtype=tf.float32)     # num_layers x [batch_size, num_units]
      hidden_outputs, hidden_states = tf.nn.dynamic_rnn(multicell, X_one_hot, dtype=tf.float32, initial_state=initial_state)
      # outputs:   [ batch_size, step_size, num_units ]
      # states:    [ num_layers, batch_size, num_units ]
      
    with tf.name_scope("output_layer"):
      hidden_outputs_flat = tf.reshape(hidden_outputs, [-1, num_units]) # [ batch_size x step_size, num_units ]
      logits_flat = layers.linear(hidden_outputs_flat, vocab_size)      # [ batch_size x step_size, vocab_size ]
      Y_hat_probs_flat = tf.nn.softmax(logits_flat)                     # [ batch_size x step_size, vocab_size ]
      Y_hat_flat = tf.cast(tf.argmax(Y_hat_probs_flat, 1), tf.uint8)    # [ batch_size x step_size ]
      
      
    with tf.name_scope("train"):
      Y_one_hot_flat = tf.reshape(Y_one_hot, [-1, vocab_size])          # [ batch_size x step_size, num_units ]
      cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot_flat, logits=logits_flat)  # [ batch_size x step_size ]
      loss = tf.reduce_mean(cost)
      optimizer = tf.train.AdamOptimizer()
      training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
      Y_flat = tf.reshape(Y, [-1])          # [ batch_size x step_size ]
      accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_flat, Y_hat_flat), tf.float32))
      likelihood_flat = tf.reduce_sum(tf.multiply(Y_hat_probs_flat, Y_one_hot_flat), 1)  # [ batch_size x step_size ]
      mean_likelihood = tf.reduce_mean(likelihood_flat)

    with tf.name_scope("summaries"):
      loss_summary = tf.summary.scalar('loss', loss)
      acc_summary = tf.summary.scalar('accuracy', accuracy)
      ml_summary = tf.summary.scalar('mean_likelihood', mean_likelihood)
      merged_summary = tf.summary.merge_all()      

    #
    # public graph nodes 
    #
    self.X = X
    self.Y = Y
    self.initial_state = initial_state
    self.hidden_states = hidden_states
    self.Y_hat_probs_flat = Y_hat_probs_flat
    self.Y_hat_flat = Y_hat_flat
    self.loss = loss
    self.training_op = training_op
    self.accuracy = accuracy
    self.mean_likelihood = mean_likelihood
    self.merged_summary = merged_summary
  
  
  def data_sequencer(self, data, verbose=True):
    _data = np.array(data)
    data_len = _data.shape[0]
    num_steps = (data_len - 1) // (self.step_size * self.batch_size)
    assert num_steps > 0
    data_len_used = self.step_size * self.batch_size * num_steps
    _xdata = np.reshape(_data[0:data_len_used], [self.batch_size, self.step_size * num_steps])
    _ydata = np.reshape(_data[1:data_len_used+1], [self.batch_size, self.step_size * num_steps])
    
    if verbose:
      range_fun = trange(num_steps)
    else:
      range_fun = range(num_steps)
    for i in range_fun:
      x = _xdata[:, i * self.step_size: (i + 1) * self.step_size]
      y = _ydata[:, i * self.step_size: (i + 1) * self.step_size]
      yield x, y

      
  def fit(self, data, restore_checkpoint=True, step_offset=None, train_logdir=None):
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not(train_logdir):
      train_logdir = "{}/{}-train".format(self.log_dir, time_string)
    save_dir = "{}/checkpoints".format(self.log_dir)
    train_file_writer = tf.summary.FileWriter(train_logdir, tf.get_default_graph())
   
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess: 
      latest_checkpoint = tf.train.latest_checkpoint(save_dir)
      if restore_checkpoint & (latest_checkpoint is not None):
        saver.restore(sess, latest_checkpoint)
      else:
        tf.global_variables_initializer().run()
      
      init_state = sess.run(self.initial_state)
      cur_step = 1
      if step_offset:
        cur_step += step_offset
      
      print('Initiating model fit...')
      print('Data length     : {}'.format(len(data)))
      print('Step size       : {}'.format(self.step_size))
      print('Batch size      : {}'.format(self.batch_size))
      print('Hidden layers   : {}'.format(self.num_layers))
      print('Units per layer : {}'.format(self.num_units))
      print('Run dir         : {}'.format(train_logdir))
    
      for x, y in self.data_sequencer(data):
        feed_dict={self.X: x, self.Y: y}
        for i, v in enumerate(self.initial_state):
            feed_dict[v] = init_state[i]
        _, last_state, summary = sess.run([self.training_op, self.hidden_states, self.merged_summary], 
                                                       feed_dict=feed_dict)
        train_file_writer.add_summary(summary, cur_step)
        
        if cur_step % 50 == 0:
            saver.save(sess, "{}/{}-{}".format(save_dir, time_string, cur_step), global_step=cur_step)

        init_state = last_state
        cur_step += 1
      
      saver.save(sess, "{}/{}-{}".format(save_dir, time_string, cur_step), global_step=cur_step)

          
  def eval(self, data):
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_logdir = "{}/{}-eval".format(self.log_dir, time_string)
    save_dir = "{}/checkpoints".format(self.log_dir)
    eval_file_writer = tf.summary.FileWriter(eval_logdir)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess: 
      latest_checkpoint = tf.train.latest_checkpoint(save_dir)
      assert (latest_checkpoint is not None)
      saver.restore(sess, latest_checkpoint)
      
      init_state = sess.run(self.initial_state)
      cur_step = 1
      
      accuracy_history = []
      mean_likelihood_history = []
      loss_history = []
      
      print('Evaluating model...')
      print('Data length     : {}'.format(len(data)))
      print('Run dir         : {}'.format(eval_logdir))

      for x, y in self.data_sequencer(data):
        feed_dict={self.X: x, self.Y: y}
        for i, v in enumerate(self.initial_state):
            feed_dict[v] = init_state[i]
        _accuracy, _mean_likelihood, _loss, last_state, summary = sess.run([self.accuracy,
                                                                     self.mean_likelihood,
                                                                     self.loss,
                                                                     self.hidden_states, 
                                                                     self.merged_summary], 
                                                                    feed_dict=feed_dict)
        eval_file_writer.add_summary(summary, cur_step)
        accuracy_history += [_accuracy]
        mean_likelihood_history += [_mean_likelihood]
        loss_history += [_loss]
        
        init_state = last_state
        cur_step += 1

      eval_accuracy = np.mean(accuracy_history)
      eval_perplexity = np.exp(np.mean(loss_history))
      eval_mean_likelihood = np.mean(mean_likelihood_history)
      
      print('Accuracy        : {:.04f}'.format(eval_accuracy))
      print('Perplexity      : {:.04f}'.format(eval_perplexity))
      print('Mean likelihood : {:.04f}'.format(eval_mean_likelihood))
      
      return eval_accuracy, eval_perplexity, eval_mean_likelihood
  
    
  def predict_log_proba(self, _X, sess=None):
    save_dir = "{}/checkpoints".format(self.log_dir)
    saver = tf.train.Saver(max_to_keep=1)

    is_local_session = False
    if sess is None:
        sess = tf.Session()
        is_local_session = True
        
    latest_checkpoint = tf.train.latest_checkpoint(save_dir)
    assert (latest_checkpoint is not None)
    saver.restore(sess, latest_checkpoint)
    
    log_probabilities = []

    for j in trange(_X.shape[0]):
        init_state = sess.run(self.initial_state)
        cur_step = 1
        loss_history = []
        for x, y in self.data_sequencer(_X[j], verbose=False):
          feed_dict={self.X: x, self.Y: y}
          for i, v in enumerate(self.initial_state):
              feed_dict[v] = init_state[i]
          _loss, last_state = sess.run([self.loss,
                                        self.hidden_states],
                                       feed_dict=feed_dict)
          loss_history += [_loss]
          init_state = last_state
        log_probabilities += [-np.mean(loss_history)]

    if is_local_session:
        sess.close()

    return np.array(log_probabilities)

