import numpy as np
import tensorflow as tf
import train_multi_rnn

import sys


train_multi_rnn.time_bin=1
train_multi_rnn.multiplier=1000
train_multi_rnn.end_symbol = -1

train_multi_rnn.keep_prob = 1.0
train_multi_rnn.pos_weight = 1 
train_multi_rnn.max_grad_norm = 5

train_multi_rnn.batch_size=1
train_multi_rnn.num_steps=1000
train_multi_rnn.max_epoch=50
train_multi_rnn.reg=0.1
train_multi_rnn.learning_rate=2.0
train_multi_rnn.cell_size = 128

if __name__ == '__main__':
    train_multi_rnn.reg=float(sys.argv[3])
    print 'Reg: {0}'.format(train_multi_rnn.reg)
    train_multi_rnn.learning_rate=float(sys.argv[4])
    print 'Learning rate: {0}'.format(train_multi_rnn.learning_rate)
    train_multi_rnn.cell_size = int(sys.argv[5])
    print 'Cell size: {0}'.format(train_multi_rnn.cell_size)
   
    suffix = sys.argv[6]
    ckt_train = 'train_best_{0}.ckt'.format(suffix)
    ckt_test = 'test_best_{0}.ckt'.format(suffix) 

    data_dir = sys.argv[2]
    with tf.device('/gpu:0'):
        train_multi_rnn.FLAGS.log_dir='/home/honglei/projects/heterlearning/log_dir'
        if sys.argv[1] == '0':    # supervised training
            m=train_multi_rnn.TrainMultiRNN(data_dir, fix_shared=False)
            m.sup_train(ckt_train)
        elif sys.argv[1] == '1':  # unsupervised training for test data
            m=train_multi_rnn.TrainMultiRNN(data_dir, fix_shared=True)
            m.partially_train('/home/honglei/projects/heterlearning/log_dir/'+ckt_train, ckt_test)
        elif sys.argv[1] == '2':   # get accuracy for test data
            m=train_multi_rnn.TrainMultiRNN(data_dir, fix_shared=True)
            m.restore('/home/honglei/projects/heterlearning/log_dir/'+ckt_test)
            acc, _, _ =  m.get_accuracy()
            print acc
        else:
            m=train_multi_rnn.TrainMultiRNN(data_dir, fix_shared=False)
            m.train('best.ckt')
