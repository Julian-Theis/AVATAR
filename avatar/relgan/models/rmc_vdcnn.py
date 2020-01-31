import tensorflow as tf
from avatar.relgan.utils.models.relational_memory import RelationalMemory
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from avatar.relgan.utils.ops import *


embedding_size = 16
filter_sizes = [3, 3, 3, 3, 3]
num_filters = [128, 256, 128, 256, 512]
num_blocks = [2, 2, 2, 2]

cnn_initializer = tf.keras.initializers.he_normal()
fc_initializer = tf.truncated_normal_initializer(stddev=0.05)


def generator(x_real, temperature, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots, head_size, num_heads,
              hidden_dim, start_token):
    start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
    output_size = mem_slots * head_size * num_heads

    g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
    g_output_unit = create_output_unit(output_size, vocab_size)

    # initial states
    init_states = gen_mem.initial_state(batch_size)

    # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
    gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                    infer_shape=True)

    def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)  # hidden_memory_tuple
        o_t = g_output_unit(mem_o_t)  # batch x vocab, logits not prob
        gumbel_t = add_gumbel(o_t)
        next_token = tf.cast(tf.argmax(gumbel_t, axis=1), tf.int32)
        x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, temperature))  # one-hot-like, [batch_size x vocab_size]
        # x_tp1 = tf.matmul(x_onehot_appr, g_embeddings)  # approximated embeddings, [batch_size x emb_dim]
        x_tp1 = tf.nn.embedding_lookup(g_embeddings, next_token)  # embeddings, [batch_size x emb_dim]
        gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, vocab_size, 1.0, 0.0),
                                                         tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
        gen_x = gen_x.write(i, next_token)  # indices, [batch_size]
        gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)
        return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv

    _, _, _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3, _4, _5: i < seq_len,
        body=_gen_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   init_states, gen_o, gen_x, gen_x_onehot_adv))

    gen_x = gen_x.stack()  # seq_len x batch_size
    gen_x = tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_len

    gen_x_onehot_adv = gen_x_onehot_adv.stack()
    gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv, perm=[1, 0, 2])  # batch_size x seq_len x vocab_size

    # ----------- pre-training for generator -----------------
    x_emb = tf.transpose(tf.nn.embedding_lookup(g_embeddings, x_real), perm=[1, 0, 2])  # seq_len x batch_size x emb_dim
    g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)

    ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len)
    ta_emb_x = ta_emb_x.unstack(x_emb)

    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)
        o_t = g_output_unit(mem_o_t)
        g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
        x_tp1 = ta_emb_x.read(i)
        return i + 1, x_tp1, h_t, g_predictions

    _, _, _, g_predictions = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < seq_len,
        body=_pretrain_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   init_states, g_predictions))

    g_predictions = tf.transpose(g_predictions.stack(),
                                 perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

    # pretraining loss
    pretrain_loss = -tf.reduce_sum(
        tf.one_hot(tf.to_int32(tf.reshape(x_real, [-1])), vocab_size, 1.0, 0.0) * tf.log(
            tf.clip_by_value(tf.reshape(g_predictions, [-1, vocab_size]), 1e-20, 1.0)
        )
    ) / (seq_len * batch_size)

    return gen_x_onehot_adv, gen_x, pretrain_loss


def discriminator(x_onehot, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, is_train=True):

    # ============= Embedding Layer =============
    d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    emb_x_re = tf.matmul(input_x_re, d_embeddings)
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim
    emb_x_expanded = tf.expand_dims(emb_x, 2)  # batch_size x seq_len x 1 x emd_dim

    # ============= First Convolution Layer =============
    with tf.variable_scope("conv-0"):
        conv0 = tf.layers.conv2d(
            emb_x_expanded,
            filters=num_filters[0],
            kernel_size=[filter_sizes[0], 1],
            kernel_initializer=cnn_initializer,
            activation=tf.nn.relu)

    # ============= Convolution Blocks =============
    conv1 = conv_block(conv0, 1, max_pool=True, is_train=is_train)

    conv2 = conv_block(conv1, 2, max_pool=False, is_train=is_train)

    # conv3 = conv_block(conv2, 3, max_pool=False, is_train=is_train)
    #
    # conv4 = conv_block(conv3, 4, max_pool=False, is_train=is_train)

    # ============= k-max Pooling =============
    h = tf.transpose(tf.squeeze(conv2), [0, 2, 1])
    top_k = tf.nn.top_k(h, k=1, sorted=False).values
    h_flat = tf.reshape(top_k, [batch_size, -1])

    # ============= Fully Connected Layers =============
    # fc1_out = tf.layers.dense(h_flat, 2048, activation=tf.nn.relu, kernel_initializer=fc_initializer)
    #
    # fc2_out = tf.layers.dense(fc1_out, 2048, activation=tf.nn.relu, kernel_initializer=fc_initializer)

    logits = tf.layers.dense(h_flat, 1, activation=None, kernel_initializer=fc_initializer)

    logits = tf.squeeze(logits, -1)  # batch_size

    return logits


def conv_block(input, i, max_pool=True, is_train=True):
    with tf.variable_scope("conv-block-%s" % i):
        # Two "conv-batch_norm-relu" layers.
        for j in range(2):
            with tf.variable_scope("conv-%s" % j):
                # convolution
                conv = tf.layers.conv2d(
                    input,
                    filters=num_filters[i],
                    kernel_size=[filter_sizes[i], 1],
                    kernel_initializer=cnn_initializer,
                    activation=None)
                # batch normalization
                conv = tf.layers.batch_normalization(conv, training=is_train)
                # relu
                conv = tf.nn.relu(conv)

        if max_pool:
            # Max pooling
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=(3, 1),
                strides=(2, 1),
                padding="SAME")
            return pool
        else:
            return conv