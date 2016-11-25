
"""
Adapted from TensorFlow's RNN sample code

Create RNN model with partially shared variables

"""

import tensorflow as tf
import rnn
import rnn_cell
import numpy as np


class SharedRNN(object):

    def __init__(self, config, reuse=False, fix_shared=False):

        with tf.variable_scope('shared_variables') as shared_scope:
            pass

        with tf.variable_scope(config.name):
            self._inputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.num_ch])
            self._outputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.num_ch])

            cell_list=[]
            for i in xrange(config.num_layers):
                # decide whether to share the variables or not based on share list
                if config.share[i]:
                    with tf.variable_scope(shared_scope, reuse=reuse):
                        with tf.variable_scope('Cell{}'.format(i)) as cell_scope:
                            lstm_cell = rnn_cell.LSTMCell(config.cell_size, cell_clip=None, num_proj=None, proj_clip=None, scope=cell_scope)
                else:
                    lstm_cell = rnn_cell.LSTMCell(config.cell_size, cell_clip=None, num_proj=None, proj_clip=None)

                if config.keep_prob < 1:
                    lstm_cell = rnn_cell.DropoutWrapper(
                        lstm_cell, output_keep_prob=config.keep_prob)
                cell_list.append(lstm_cell)

            cell = rnn_cell.MultiRNNCell(cell_list)

            self._initial_state = cell.zero_state(config.batch_size, tf.float32)

            #input_w = tf.get_variable('input_w', [config.num_ch, config.cell_size], initializer = orthogonal_initializer()) # use orthogonal initializer
            input_w = tf.get_variable('input_w', [config.num_ch, config.cell_size], initializer = tf.random_uniform_initializer(-1,1)) # use random initializer
            #input_b = tf.get_variable('input_b', [config.cell_size],initializer=tf.constant_initializer(0))
            # input_b = 0
            self._rnn_input = tf.reshape(tf.matmul(tf.reshape(self._inputs, (config.batch_size * config.num_steps, config.num_ch)), input_w), (config.batch_size, config.num_steps, config.cell_size))
            if config.keep_prob < 1:
                self._rnn_input = tf.nn.dropout(self._rnn_input, config.keep_prob)

            lengths=tf.fill([config.batch_size],config.num_steps)
            self._rnn_output, state = rnn.dynamic_rnn(cell, self._rnn_input, lengths, initial_state=self._initial_state)

            output_w = tf.get_variable('output_w', [config.cell_size, config.num_ch], initializer = tf.random_uniform_initializer(-1,1))
            #output_w=tf.transpose(input_w)
            #output_b = tf.get_variable('output_b', [config.num_ch],initializer=tf.constant_initializer(0))
            #output_b = 0
            logits = tf.reshape(tf.matmul(tf.reshape(self._rnn_output, (config.batch_size*config.num_steps,config.cell_size)),output_w), (config.batch_size,config.num_steps,config.num_ch))
            # batch_loss = tf.nn.weighted_cross_entropy_with_logits(logits,self._outputs,config.pos_weight)  # when only unsupervised training is used
            batch_loss = tf.nn.weighted_cross_entropy_with_logits(tf.slice(logits,[0,0,0],[-1, config.num_steps-1, -1]), tf.slice(self._outputs,[0,0,0],[-1, config.num_steps-1, -1]),config.pos_weight)     # when supervised training is used, there is an end vector

            # L1 identity regularization
            # identity_loss = tf.reduce_mean(tf.abs(tf.matmul(input_w, output_w) - tf.constant(np.identity(config.num_ch), dtype = tf.float32)))
            # L2 identity regularization
            # identity_loss = tf.nn.l2_loss(tf.matmul(input_w, output_w) - tf.constant(np.identity(config.num_ch), dtype = tf.float32))
            
            ## final loss
            
            # with identity regularization
            # self._loss = tf.reduce_mean(batch_loss) + config.reg * identity_loss
           
            # with L1 regularization
            #self._loss = tf.reduce_mean(batch_loss)+config.reg*tf.reduce_sum(tf.abs(input_w))+config.reg*tf.reduce_sum(tf.abs(output_w))
            
            # with L2 regularization
            self._loss = tf.reduce_mean(batch_loss) + config.reg*tf.nn.l2_loss(input_w) +  config.reg*tf.nn.l2_loss(output_w) 
            
            ## final state
            self._final_state = state[0].h

            ## mean pooling over time 
            self._mean_state = tf.reduce_mean(self._rnn_input, 1)

            ## predicts
            self._predicts=tf.sigmoid(logits)
            
            ## when using supervised training, add softmax loss
            with tf.variable_scope(shared_scope, reuse=reuse):
                softmax_w = tf.get_variable('softmax_w', [config.cell_size,config.num_classes])
            # use final state for classification
            # softmax_logits=tf.matmul(self._final_state, softmax_w)
            # use mean state for classification
            softmax_logits=tf.matmul(self._mean_state, softmax_w)
            self._label_predicts=tf.nn.softmax(softmax_logits)
            if config.recording_label!=-1:
                r_labels = tf.fill([config.batch_size], config.recording_label)
                softmax_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(softmax_logits, r_labels)
                self._sup_loss=self._loss+tf.reduce_mean(softmax_loss) + config.reg*tf.nn.l2_loss(softmax_w)

            # decide which variables to train
            if fix_shared:
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                #tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, shared_scope.name)
            else:
                tvars = tf.trainable_variables()

            print 'Training variables:'
            print '\n'.join([x.name for x in tvars])
            
            # learning rate decay
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = config.learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             decay_steps=20, decay_rate=0.96, staircase=True)
            
            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.001)
            #optimizer = tf.train.MomentumOptimizer(config.learning_rate,0.9)
            #optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            
            #self._train_op = optimizer.minimize(self._loss, var_list=tvars)

            # manual clipping
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                              config.max_grad_norm)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            ## when using supervised training, add supervised training op
            if config.recording_label != -1:
                sup_grads, _ = tf.clip_by_global_norm(tf.gradients(self._sup_loss, tvars),
                                                  config.max_grad_norm)

                self._sup_train_op = optimizer.apply_gradients(zip(sup_grads, tvars), global_step=global_step)

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
    def sup_loss(self):
        return self._sup_loss
    
    @property
    def final_state(self):
        return self._final_state

    @property
    def predicts(self):
        return self._predicts
 
    @property
    def label_predicts(self):
        return self._label_predicts

    @property
    def train_op(self):
        return self._train_op

    @property
    def sup_train_op(self):
        return self._sup_train_op

    @property
    def mean_state(self):
        return self._mean_state
    
def orthogonal_initializer(scale=1.0):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer

class SharedRNNConfig(object):
    """configurations for sharedRNN"""
    num_layers = 1
    share=[True]
    cell_size = 8
    
    batch_size= 10
    num_steps = 100
    num_ch = 10
    name=None
    num_classes=3
    recording_label=-1
    
    keep_prob = 1.0
    reg=0.01
    pos_weight = 100 # positive weight for weighted loss function
    max_grad_norm = 2
    learning_rate=2.0


