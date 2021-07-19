from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mae_loss
from model.ddcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, is_training, batch_size,  **model_kwargs):

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))


        self._inputs = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        self._labels = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')
        self._adj_mx = tf.compat.v1.sparse_placeholder(tf.float32, name='adj_mx')


        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))
        cell = DCGRUCell(rnn_units, self._adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units, self._adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)
        encoding_cells = [cell] * num_rnn_layers
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(decoding_cells, state_is_tuple=True)


        global_step = tf.compat.v1.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.compat.v1.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels.insert(0, GO_SYMBOL)

            def loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random.uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(pred=tf.less(c, threshold), true_fn=lambda: labels[i], false_fn=lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                return result

            _, enc_state = tf.compat.v1.nn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            
            state = enc_state
            outputs = []
            prev = None
            for i, inp in enumerate(labels):
                if loop_function is not None and prev is not None:
                    inp = loop_function(prev, i)
                output, state = decoding_cells(inp, state)
                outputs.append(output)
                if loop_function is not None:
                    prev = output
            #outputs, state
            #print('outputs', len(outputs))

        # Project the output to output_dim.
        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.compat.v1.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @property
    def adj_mx(self):
        return self._adj_mx
        