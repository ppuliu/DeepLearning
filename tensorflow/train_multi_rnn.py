import numpy as np
import time
import os
import sys
from sklearn.metrics import roc_auc_score

from multi_rnn import *


batch_size=1
num_steps=100
max_epoch=100

class TrainMultiRNN(object):

    def __init__(self, file_dir, fix_shared=False):

        print("Reading data")
        self._file_dir=file_dir
        self._data_list, self._configs=self.read_files(file_dir)
        self.resample_data(train_ratio=0.8,val_ratio=0.1)

        print("Building model")
        # build the model
        initializer = tf.random_uniform_initializer(-1, 1)
        with tf.variable_scope("model", initializer=initializer):
            self._multi_rnn = MultiRNN(self._configs, fix_shared)

        print("Starting session")
        session=tf.Session()
        session.run(tf.initialize_all_variables())
        self._sess=session

    def read_files(self, file_dir):
        """

        Args:
            file_dir:

        Returns:
            data_list: list of data num_steps x num_ch
            configs: list of rnn configurations
        """
        if not os.path.isdir(file_dir):
            raise ValueError("input direcotry doesn't exist!")
        data_list=[]
        configs=[]
        for file_name in os.listdir(file_dir):
            file_path=os.path.join(file_dir,file_name)
            data, config=self.read_data(file_path)
            data_list.append(data)
            configs.append(config)

        return data_list, configs

    def read_data(self, file_path):
        """

        Args:
            file_path:

        Returns:
            data: num_steps x num_ch array
            config:  rnn configuration

        """
        data = np.genfromtxt(file_path, delimiter=',')
        _, num_ch = data.shape
        config=self.get_config(file_path,num_ch)

        return data, config

    def get_config(self, name, num_ch):

        config=SharedRNNConfig()
        config.name=name
        config.num_ch=num_ch
        config.batch_size=batch_size
        config.num_steps=num_steps

        return config

    def resample_data(self, train_ratio=0.8, val_ratio=0.1):

        self._train_data_list=[]
        self._val_data_list=[]
        self._test_data_list=[]

        for data in self._data_list:
            data_steps, data_num_ch = data.shape
            train_len = int(data_steps * train_ratio)
            val_len = int(data_steps * val_ratio)

            self._train_data_list.append(data[0:train_len,:])
            self._val_data_list.append(data[train_len:train_len + val_len, :])
            self._test_data_list.append(data[train_len + val_len:-1, :])

    def partially_train(self, checkpoint_file):

        saver = tf.train.Saver(var_list=self._multi_rnn.get_fixed_variables())
        saver.restore(self._sess, checkpoint_file)

        self.train()


    def train(self):
        """
        run the training process

        Returns:

        """
        try:
            session=self._sess

            for i in range(max_epoch):
                train_perplexity=self.run_epoch_training(session,self._train_data_list,verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                train_roc_auc = self.run_eval(session, self._train_data_list, verbose=True)
                print("Epoch: %d Train ROC-AUC: %.3f" % (i + 1, train_roc_auc))
                val_roc_auc = self.run_eval(session, self._val_data_list, verbose=True)
                print("Epoch: %d Valid ROC-AUC: %.3f" % (i + 1, val_roc_auc))

            test_roc_auc = self.run_eval(session, self._test_data_list, verbose=True)
            print("Test ROC-AUC: %.3f" % test_roc_auc)

        except KeyboardInterrupt:
            print("WARNING: User interrupted program.")

            finalizeAndSave = raw_input("Do you want to save the latest data? [y/n]")
            if finalizeAndSave != 'n':
                save_path= raw_input("Save results to: ")
                print("Saving latest results.")
                self.save(save_path)
            else:
                print("Results deleted.")

            self.close()

    def eval(self):

        session = self._sess

        train_roc_auc = self.run_eval(session, self._train_data_list, verbose=True)
        print("Train ROC-AUC: %.3f" % (train_roc_auc))
        val_roc_auc = self.run_eval(session, self._val_data_list, verbose=True)
        print("Valid ROC-AUC: %.3f" % (val_roc_auc))

        test_roc_auc = self.run_eval(session, self._test_data_list, verbose=True)
        print("Test ROC-AUC: %.3f" % test_roc_auc)


    def run_epoch_training(self, session, data_list, verbose=False):
        """Go through all the datasets once"""
        num_datasets=len(data_list)

        # calculate the maximum epoch_size for all the datasets
        epoch_size=0
        for data in data_list:
            total_steps, _ = data.shape
            curr_size=(total_steps - num_steps) // batch_size
            epoch_size=max(epoch_size, curr_size)

        start_time = time.time()
        loss = 0.0
        iters = 0

        for step in xrange(epoch_size):
            loss_list=[]
            for data_index in xrange(num_datasets):
                data=data_list[data_index]
                data_steps, data_num_ch = data.shape
                start = step * num_steps * batch_size
                end = (step + 1) * num_steps * batch_size
                if end>=data_steps:
                    continue

                # get a batch of training data
                x = data[start:end, :].T.reshape((batch_size, data_num_ch, num_steps), order='C').transpose((0, 2, 1))
                y = data[start+1:end+1, :].T.reshape((batch_size, data_num_ch, num_steps), order='C').transpose(
                    (0, 2, 1))

                # get tensorflow nodes
                train_op=self._multi_rnn.get_train_op(data_index)
                loss_op=self._multi_rnn.get_loss(data_index)
                input_placeholder=self._multi_rnn.get_input_placeholder(data_index)
                output_placeholder=self._multi_rnn.get_output_placeholder(data_index)

                _, curr_loss = session.run([train_op,loss_op],
                                             {input_placeholder: x,
                                              output_placeholder: y})
                loss_list.append(curr_loss)
                loss += curr_loss
                iters += 1

            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f batches/sec" %
                    (step * 1.0 / epoch_size, np.exp(loss / iters),
                     iters / (time.time() - start_time)))

        return np.exp(loss / iters)

    def run_eval(self, session, data_list, verbose=False):

        num_datasets = len(data_list)

        # calculate the maximum epoch_size for all the datasets
        epoch_size = 0
        for data in data_list:
            total_steps, _ = data.shape
            curr_size = (total_steps - num_steps) // batch_size
            epoch_size = max(epoch_size, curr_size)

        iters = 0
        y_true = np.array([])
        y_predict=np.array([])

        for step in xrange(epoch_size):

            for data_index in xrange(num_datasets):
                data = data_list[data_index]
                data_steps, data_num_ch = data.shape
                start = step * num_steps * batch_size
                end = (step + 1) * num_steps * batch_size
                if end >= data_steps:
                    continue

                # get a batch of training data
                x = data[start:end, :].T.reshape((batch_size, data_num_ch, num_steps), order='C').transpose((0, 2, 1))
                y = data[start + 1:end + 1, :].T.reshape((batch_size, data_num_ch, num_steps), order='C').transpose(
                    (0, 2, 1))

                # get tensorflow nodes
                predict_op = self._multi_rnn.get_predict_op(data_index)
                input_placeholder = self._multi_rnn.get_input_placeholder(data_index)
                output_placeholder = self._multi_rnn.get_output_placeholder(data_index)

                predicts = np.squeeze(np.array(session.run([predict_op],
                                           {input_placeholder: x,
                                            output_placeholder: y})))


                iters += 1

                y_true= np.append(y_true, y[:, 50:-1, :])
                y_predict=np.append(y_predict, predicts[:, 50:-1, :])

        return roc_auc_score(y_true, y_predict)

    def restore(self, checkpoint_file):

        saver = tf.train.Saver()
        saver.restore(self._sess, checkpoint_file)

    def save(self, path):
        saver = tf.train.Saver()
        saver.Save(self._sess, path)

    def close(self):

        self._sess.close()

if __name__ == "__main__":
    file_dir=sys.argv[1]

    m=TrainMultiRNN(file_dir)
    m.train()