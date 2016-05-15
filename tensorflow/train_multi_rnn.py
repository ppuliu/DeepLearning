import numpy as np
import time
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score

from multi_rnn import *

batch_size=1
num_steps=100
max_epoch=100

flags = tf.flags
flags.DEFINE_string(
    "log_dir", "./logdir","log directory")

FLAGS = flags.FLAGS

class TrainMultiRNN(object):

    def __init__(self, file_dir, fix_shared=False):

        print("Reading data")
        self._file_dir=file_dir
        self._data_list, self._configs=self.read_files(file_dir)
        self.resample_data(train_ratio=0.8,val_ratio=0.1)

        print("Building model")
        # build the model
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", initializer=initializer):
            self._multi_rnn = MultiRNN(self._configs, fix_shared)

        print("Starting session")
        session=tf.Session()
        session.run(tf.initialize_all_variables())
        self._sess=session

        # set up summaries
        self.add_summaries()
        self._summary_op=tf.merge_all_summaries()
        self._summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, self._sess.graph)

    def add_summaries(self):
        var_list=tf.all_variables()
        for var in var_list:
            # if len(var.get_shape())>0:
            #     tf.histogram_summary(var.name, var)
            # else:
            #     tf.scalar_summary(var.name, var)
            if 'share' in var.name:
                tf.scalar_summary(var.name+'_max', tf.reduce_max(var))
                tf.scalar_summary(var.name+'_min', tf.reduce_min(var))
                tf.scalar_summary(var.name+'_mean', tf.reduce_mean(var))
                tf.histogram_summary(var.name, var)
            print var.name

        for i in xrange(len(self._multi_rnn.loss_ops)):
            tf.scalar_summary('loss_{}'.format(i+1), self._multi_rnn.get_loss(i))

    def get_value_of_variable(self, name):

        var_list=[v for v in tf.all_variables() if v.name==name]
        if len(var_list)==0:
            return None
        else:
            return self._sess.run(var_list[0])

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
        data = np.genfromtxt(file_path, dtype=int, delimiter=',')
        if len(data.shape)==1:
            data=np.expand_dims(data, axis=1)
        _, num_ch = data.shape
        config=self.get_config(os.path.basename(file_path),num_ch)

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

        print self._train_data_list[0].shape

    def partially_train(self, checkpoint_file):

        saver = tf.train.Saver(var_list=self._multi_rnn.fixed_variables)
        saver.restore(self._sess, checkpoint_file)

        self.train()


    def train(self):
        """
        run the training process

        Returns:

        """
        try:
            session=self._sess
            auc_histroy=[0]
            for i in range(max_epoch):
                train_loss=self.run_epoch_training(session,self._train_data_list,i,verbose=True)
                print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))

                roc_auc, pr_auc = self.run_eval(session, self._train_data_list, verbose=True)
                print("Epoch: %d Train ROC-AUC: %.3f, PR-AUC: %.3f" % (i + 1, roc_auc, pr_auc))
                if roc_auc>=max(auc_histroy):
                    self.save(os.path.join(FLAGS.log_dir,'best.ckt'))
                auc_histroy.append(roc_auc)
                roc_auc, pr_auc = self.run_eval(session, self._val_data_list, verbose=True)
                print("Epoch: %d Valid ROC-AUC: %.3f, PR-AUC: %.3f" % (i + 1, roc_auc, pr_auc))


            roc_auc, pr_auc = self.run_eval(session, self._test_data_list, verbose=True)
            print("Test ROC-AUC: %.3f, PR-AUC: %.3f" % (roc_auc, pr_auc))

        except KeyboardInterrupt:
            print("WARNING: User interrupted program.")

            finalizeAndSave = raw_input("Do you want to save the latest data? [y/n]")
            if finalizeAndSave != 'n':
                save_path = raw_input("Save results to: ")
                self.save(save_path)
            else:
                print("Results deleted.")


    def get_eval_data(self):

        session = self._sess

        return self.run_predicts(session, self._val_data_list, verbose=True)

    def get_train_data(self):

        session = self._sess

        return self.run_predicts(session, self._train_data_list, verbose=True)

    def run_predicts(self, session, data_list, verbose=False):

        num_datasets = len(data_list)
        origin_data_list=[]
        predicted_data_list=[]

        # calculate the maximum epoch_size for all the datasets
        epoch_size = 0
        for data in data_list:
            total_steps, data_num_ch = data.shape
            curr_size = (total_steps // num_steps) // batch_size
            epoch_size = max(epoch_size, curr_size)
            origin_data_list.append(np.array([]).reshape(0,data_num_ch))
            predicted_data_list.append(np.array([]).reshape(0,data_num_ch))


        iters = 0

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

                predicts = np.array(session.run([predict_op],
                                                           {input_placeholder: x,
                                                            output_placeholder: y}))

                iters += 1

                origin_data_list[data_index] = np.append(origin_data_list[data_index], data[start + 1:end + 1, :],
                                                         axis=0)
                predicted_data_list[data_index] = np.append(predicted_data_list[data_index],
                                                            predicts.reshape(batch_size*num_steps, data_num_ch),axis=0)


        return origin_data_list, predicted_data_list

    def eval(self):

        session = self._sess

        roc_auc, pr_auc = self.run_eval(session, self._train_data_list, verbose=True)
        print("Train ROC-AUC: %.3f, PR-AUC: %.3f" % (roc_auc, pr_auc))

        roc_auc, pr_auc = self.run_eval(session, self._val_data_list, verbose=True)
        print("Valid ROC-AUC: %.3f, PR-AUC: %.3f" % (roc_auc, pr_auc))

        roc_auc, pr_auc = self.run_eval(session, self._test_data_list, verbose=True)
        print("Test ROC-AUC: %.3f, PR-AUC: %.3f" % (roc_auc, pr_auc))


    def run_epoch_training(self, session, data_list, num_epoch, verbose=False):
        """Go through all the datasets once"""
        num_datasets=len(data_list)

        # calculate the maximum epoch_size for all the datasets
        epoch_size=0
        for data in data_list:
            total_steps, _ = data.shape
            curr_size=(total_steps // num_steps) // batch_size
            print total_steps
            print num_steps
            epoch_size=max(epoch_size, curr_size)
        print epoch_size

        start_time = time.time()
        loss = 0.0
        iters = 0

        all_data_feed = {}
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

                feed_dict={input_placeholder: x, output_placeholder: y}
                _, curr_loss = session.run([train_op,loss_op], feed_dict)
                loss_list.append(curr_loss)
                loss += curr_loss
                iters += 1

                # gather placeholder data
                all_data_feed.update(feed_dict)

            if verbose and step % (epoch_size // 10) == 0:
                print("%d/%d loss: %.3f speed: %.0f batches/sec" %
                    (step, epoch_size, loss / iters,
                     iters / (time.time() - start_time)))

                # write summaries
                summary_str = session.run(self._summary_op, all_data_feed)
                self._summary_writer.add_summary(summary_str, iters+num_epoch*epoch_size*num_datasets)
                self._summary_writer.flush()

        return loss / iters

    def run_eval(self, session, data_list, verbose=False):

        num_datasets = len(data_list)

        # calculate the maximum epoch_size for all the datasets
        epoch_size = 0
        for data in data_list:
            total_steps, _ = data.shape
            curr_size = (total_steps // num_steps) // batch_size
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

                y_true= np.append(y_true, y.flatten())
                y_predict=np.append(y_predict, predicts.flatten())

        roc_auc=roc_auc_score(y_true, y_predict)

        pr_auc=average_precision_score(y_true, y_predict)
        #pr_auc=fbeta_score(y_true, y_predict>0.5, beta=1)

        return roc_auc, pr_auc

    def restore(self, checkpoint_file):

        saver = tf.train.Saver()
        saver.restore(self._sess, checkpoint_file)

    def save(self, path):
        print("Saving latest results.")
        saver = tf.train.Saver()
        saver.save(self._sess, path)

    def close_and_save(self):
        save_path = raw_input("Save results to: ")
        self.save(os.path.join(FLAGS.log_dir, save_path))
        self._summary_writer.close()
        self._sess.close()

    def close(self):
        self._summary_writer.close()
        self._sess.close()

if __name__ == "__main__":
    file_dir=sys.argv[1]
    FLAGS.log_dir=sys.argv[2]

    m=TrainMultiRNN(file_dir, fix_shared=True)
    #m.train()
    #m.partially_train('/home/honglei/projects/neural_network/log_dir/1_layer_150_neuron_r_2_875.ckt')
    #m.restore(os.path.join(FLAGS.log_dir,'1_layer_150_neuron_r_1_889.ckt'))
    #print '---'
    #m.eval()
