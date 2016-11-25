
from partially_shared_rnn import *

class MultiRNN(object):

    def __init__(self, configs, fix_shared=False):
        
        self._train_ops=[]
        self._predicts=[]        
        self._models=[]
        self._loss=[]
        self._final_state=[]
        self._mean_state=[]
        
        self._label_predicts=[]
        self._sup_loss=[]
        self._sup_train_ops = []
        for i in xrange(len(configs)):
            config=configs[i]
            print '\nBuilding model for {0} with label {1}'.format(config.name, config.recording_label)
            if i==0:
                shared_rnn=SharedRNN(config, reuse=False, fix_shared=fix_shared)
            else:
                shared_rnn = SharedRNN(config, reuse=True, fix_shared=fix_shared)
            self._models.append(shared_rnn)
            self._predicts.append(shared_rnn.predicts)
            self._loss.append(shared_rnn.loss)
            self._train_ops.append(shared_rnn.train_op)
            self._final_state.append(shared_rnn.final_state)
            self._mean_state.append(shared_rnn._mean_state)

            # supervised learning
            self._label_predicts.append(shared_rnn.label_predicts)
            self._sup_loss.append(shared_rnn.sup_loss)
            if config.recording_label != -1:
                self._sup_train_ops.append(shared_rnn.sup_train_op)

        with tf.variable_scope('shared_variables') as shared_scope:
            self._shared_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, shared_scope.name)
            print '\nShared variables:'
            print '\n'.join([x.name for x in self._shared_variables])

    def get_input_placeholder(self, index):
        return self._models[index].inputs

    def get_output_placeholder(self, index):
        return self._models[index].outputs

    def get_train_op(self, index):
        return self._train_ops[index]

    def get_predict_op(self, index):
        return self._predicts[index]

    def get_loss(self, index):
        return self._loss[index]
    
    def get_final_state(self, index):
        return self._final_state[index]
    
    def get_mean_state(self, index):
        return self._mean_state[index]
    
    def get_label_predict_op(self, index):
        return self._label_predicts[index]
    
    def get_sup_loss(self, index):
        return self._sup_loss[index]
    
    def get_sup_train_op(self, index):
        return self._sup_train_ops[index]

    @property
    def shared_variables(self):
        return  self._shared_variables

    @property
    def models(self):
        return self._models

    @property
    def predicts(self):
        return self._predicts

    @property
    def train_ops(self):
        return self._train_ops

    @property
    def loss_ops(self):
        return self._loss

    @property
    def sup_train_ops(self):
        return self._sup_train_ops


