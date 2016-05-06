
from partially_shared_fcnet import *

class MultiFCNet(object):

    def __init__(self, configs, fix_shared=False):

        self._train_ops=[]
        self._predicts=[]
        self._models=[]
        self._loss=[]
        for i in xrange(len(configs)):
            config=configs[i]
            if i==0:
                shared_fcnet=SharedFCNet(config, reuse=False, fix_shared=fix_shared)
            else:
                shared_fcnet = SharedFCNet(config, reuse=True, fix_shared=fix_shared)
            self._models.append(shared_fcnet)
            self._predicts.append(shared_fcnet.predicts)
            self._loss.append(shared_fcnet.loss)
            self._train_ops.append(shared_fcnet.train_op)

        with tf.variable_scope('shared_variables') as shared_scope:
            self._fixed_variables=tf.get_collection(tf.GraphKeys.VARIABLES, shared_scope.name)
            print 'Fixed variables:', [x.name for x in self._fixed_variables]

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

    @property
    def fixed_variables(self):
        return  self._fixed_variables

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


