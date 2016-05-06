
"""
Adapted from TensorFlow's RNN sample code

Create RNN model with partially shared variables

"""

import tensorflow as tf
import rnn
import rnn_cell


class SharedFCNet(object):

    def __init__(self, config, reuse=False, fix_shared=False):

        if len(config.hidden_sizes)<1:
            raise ValueError('At least one hidden layer is needed.')

        with tf.variable_scope('shared_variables') as shared_scope:
            pass

        with tf.variable_scope(config.name):
            self._inputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.input_size])
            self._outputs = tf.placeholder(tf.float32, [config.batch_size, config.num_steps, config.output_size])

            sizes=config.hidden_sizes
            sizes.insert(0,config.input_size)
            sizes.append(config.output_size)
            h=None
            for i in xrange(len(sizes)-1):
                # decide whether to share the variables or not based on share list
                if config.share[i]:
                    with tf.variable_scope(shared_scope, reuse=reuse):
                        with tf.variable_scope('Layer{}'.format(i+1)):
                            w=tf.get_variable('weights',shape=[sizes[i],sizes[i+1]],trainable=(not fix_shared))
                            #b=tf.get_variable('bias',shape=[sizes[i+1]],trainable=(not fix_shared),initializer=tf.constant_initializer(0))
                            b=0
                else:
                    with tf.variable_scope('Layer{}'.format(i + 1)):
                        w = tf.get_variable('weights', shape=[sizes[i], sizes[i + 1]], trainable=(not fix_shared))
                        #b = tf.get_variable('bias', shape=[sizes[i + 1]], trainable=(not fix_shared),initializer=tf.constant_initializer(0))
                        b=0
                # input layer
                if i==0:
                    h = tf.nn.tanh(tf.matmul(tf.reshape(self._inputs, (config.batch_size * config.num_steps, config.input_size)), w)+b)
                # output layer
                elif i==len(sizes)-2:
                    h = tf.nn.sigmoid(tf.matmul(h, w)+b)
                # hidden layers
                else:
                    h = tf.matmul(h, w) + b

                if config.keep_prob < 1:
                    h = tf.nn.dropout(h, config.keep_prob)

            logits = tf.reshape(h, (config.batch_size,config.num_steps,config.output_size))
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,self._outputs)

            self._loss = tf.reduce_mean(batch_loss)

            self._predicts=tf.sigmoid(logits)

            # decide which variables to train
            tvars = tf.trainable_variables()
            print 'Training variables:', [x.name for x in tvars]

            optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
            self._train_op = optimizer.minimize(self._loss, var_list=tvars)

            # manual clipping
            #optimizer = tf.train.MomentumOptimizer(config.learning_rate,0.9)
            #grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
            #self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


    @property
    def loss(self):
        return self._loss

    @property
    def predicts(self):
        return self._predicts

    @property
    def train_op(self):
        return self._train_op

class SharedFCNetConfig(object):
    """configurations for sharedRNN"""
    hidden_sizes = [10]
    keep_prob = 1.0
    batch_size= 10
    num_steps = 100
    input_size = 10
    output_size = 10
    share=[False, False]
    name=None
    learning_rate=0.1
    max_grad_norm = 5
    reg=0.0
