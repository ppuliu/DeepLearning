
from partially_shared_rnn import *

class MultiRNN(object):

    def __init__(self, configs, fix_shared=False):

        self._train_ops=[]
        self._predicts=[]
        self._models=[]
        self._loss=[]
        for config in configs:
            shared_rnn=SharedRNN(config, fix_shared)
            self._models.append(shared_rnn)
            self._predicts.append(shared_rnn.predicts)
            self._loss.append(shared_rnn.loss)
            self._train_ops.append(shared_rnn.train_op)

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
    def models(self):
        return self._models

    @property
    def predicts(self):
        return self._predicts

    @property
    def train_ops(self):
        return self._train_ops


