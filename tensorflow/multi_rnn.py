
from partially_shared_rnn import *

class MultiRNN(object):

    def __init__(self, configs):
        num_models=len(configs)
        self._data_index=tf.placeholder(tf.int32, [1])
        loss_list=[]
        predict_list=[]
        self._models=[]
        for config in configs:
            shared_rnn=SharedRNN(config)
            self._models.append(shared_rnn)
            loss_list.append(shared_rnn.loss)
            predict_list.append(shared_rnn.predicts)

        one_hot_index = tf.one_hot(self._data_index, num_models, 1, 0)
        self._loss = tf.matmul(tf.transpose(tf.pack(loss_list)), one_hot_index)
        self._predicts = tf.slice(tf.concat(0,predict_list),

    def input_placeholder(self, index):
        return self._models[index].inputs

    def output_placeholder(self, index):
        return self._models[index].outputs

    @property
    def models(self):
        return self._models

    @property
    def loss(self):
        return self._loss

class MultiRNNConfig(object):
    learning_rate = 1.0
    max_grad_norm = 5
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000



def run_epoch(session, m, data_in, data_out, eval_op, verbose=False):
    """Runs the model on the given data."""
    total_steps, neuronN = data_in.shape
    epoch_size = (total_steps // m.num_steps) // m.batch_size
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()

    for step in xrange(epoch_size):
        start = step * m.num_steps * m.batch_size
        end = (step + 1) * m.num_steps * m.batch_size
        x = data_in[start:end, :].T.reshape((m.batch_size, neuronN, m.num_steps), order='C').transpose((0, 2, 1))
        y = data_out[start:end, :].T.reshape((m.batch_size, neuronN, m.num_steps), order='C').transpose((0, 2, 1))
        print(x.shape, y.shape)
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += 1

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def read_data(path):
    recordings = np.genfromtxt(path, delimiter=',')
    return recordings


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = read_data(FLAGS.data_path)
    total_steps, neuronN = raw_data.shape

    train_len = int(total_steps * 0.8)
    val_len = int(total_steps * 0.1)
    test_len = val_len

    train_in = raw_data[0:train_len, :]
    train_out = raw_data[1:train_len + 1, :]
    val_in = raw_data[train_len:train_len + val_len, :]
    val_out = raw_data[train_len + 1:train_len + val_len + 1, :]
    test_in = raw_data[train_len + val_len:-1, :]
    test_out = raw_data[train_len + val_len + 1:, :]

    config = SharedRNNConfig()
    eval_config = get_config()
    config.num_ch = neuronN
    eval_config.num_ch = neuronN

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LSTM(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = LSTM(is_training=False, config=config)
            mtest = LSTM(is_training=False, config=eval_config)

        print('aaaaaa')

        tf.initialize_all_variables().run()

        for i in range(config.max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
