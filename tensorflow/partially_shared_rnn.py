
"""
Adapted from TensorFlow's RNN sample code

Create RNN model with partially shared variables

"""

import tensorflow as tf
from tensorflow.python.ops import rnn

class SharedRNN(object):

    def __init__(self, config, fix_shared=False):

        shared_scope=tf.variable_scope('shared_variables')

        with tf.variable_scope(config.name):
            self._inputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.num_ch])
            self._outputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.num_ch])

            cell_list=[]
            for i in xrange(config.num_layers):
                # decide whether to share the variables or not based on share list
                if config.share[i]:
                    cell_scope=tf.variable_scope(shared_scope,reuse=True)
                else:
                    cell_scope = tf.variable_scope('cell_layer_{0}'.format(i))
                with cell_scope:
                    # Slightly better results can be obtained with forget gate biases
                    # initialized to 1 but the hyperparameters of the model would need to be
                    # different than reported in the paper.
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.cell_size, forget_bias=0.0)
                    if config.keep_prob < 1:
                        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                            lstm_cell, output_keep_prob=config.keep_prob)
                    cell_list.append(lstm_cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

            self._initial_state = cell.zero_state(config.batch_size, tf.float32)

            input_w = tf.get_variable('input_w', [config.num_ch, config.cell_size])
            self._rnn_input = tf.reshape(tf.matmul(tf.reshape(self._inputs, (config.batch_size * config.num_steps, config.num_ch)), input_w),
                                (config.batch_size, config.num_steps, config.cell_size))
            if config.keep_prob < 1:
                self._rnn_input = tf.nn.dropout(self._rnn_input, config.keep_prob)

            lengths=tf.fill([config.batch_size],config.num_steps)
            self._rnn_output, state = rnn.dynamic_rnn(cell, self._rnn_input, lengths, initial_state=self._initial_state)

            output_w = tf.get_variable('output_w', [config.cell_size, config.num_ch])
            logits = tf.reshape(tf.matmul(tf.reshape(self._rnn_output, (config.batch_size*config.num_steps,config.cell_size)),output_w),
                              (config.batch_size,config.num_steps,config.num_ch))
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,self._outputs)

            # logits = tf.matmul(output, output_w)
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(logits, [-1]), tf.reshape(self._targets, [-1]))

            self._loss = tf.reduce_mean(batch_loss)
            self._final_state = state

            self._predicts=tf.sigmoid(logits)

            # decide which variables to train
            if fix_shared:
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.name)
            else:
                tvars = tf.trainable_variables()

            optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
            self._train_op = optimizer.minimize(self._loss, var_list=tvars)

    def get_fixed_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'shared_variables')

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def loss(self):
        return self._loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def predicts(self):
        return self._predicts

    @property
    def train_op(self):
        return self._train_op

class SharedRNNConfig(object):
    """configurations for sharedRNN"""
    num_layers = 1
    cell_size = 120
    keep_prob = 1.0
    batch_size= 10
    num_steps = 100
    num_ch = 10
    share=[False]*num_layers
    name=None
    learning_rate=0.1
