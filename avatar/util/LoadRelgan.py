import tensorflow as tf
import numpy as np
import json
import os
import time
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

class LoadRelgan():
    def __init__(self, system, suffix, job):
        if DATA_PATH is None:
            path = os.path.join(WORK_PATH, "data", "avatar","sgans", system, str(job), "tf_logs", "ckpt")
        else:
            path = os.path.join(DATA_PATH, "avatar", "sgans", system, str(job), "tf_logs", "ckpt")

        saver = tf.train.import_meta_graph(os.path.join(path, system + ".adv_model-" + str(suffix) + ".meta"))

        self.data_model = os.path.join(path, system + ".adv_model-" + str(suffix))

        self.disc_in_x = tf.get_default_graph().get_tensor_by_name('x_real:0')

        self.disc_out = tf.get_default_graph().get_tensor_by_name('prob_discriminator/prob_disc_out:0')

        self.gen_out = tf.get_default_graph().get_tensor_by_name('generator/gen_x_output:0')
        self.batch_size = self.gen_out.shape[0]

        self.sess = tf.Session()
        saver.restore(self.sess, self.data_model)

        with open(os.path.join(path, "iw_dict_" + system + ".json")) as handle:
            self.iw = json.loads(handle.read())

        self.wi_dict = {v: k for k, v in self.iw.items()}
        highest = 0
        for k, _ in self.iw.items():
            if int(k) > highest:
                highest = int(k)
        self.end_token = highest + 1
        self.seq_len = self.disc_in_x.shape[1]

    def generate(self, n_samples, unique=True):
        samples = None
        if n_samples > 0:
            n_batches = int(n_samples // self.batch_size)

            last_batch = n_samples % self.batch_size
            start = time.time()
            for i in range(n_batches):
                gen = self.sess.run(self.gen_out)

                if samples is None:
                    samples = gen
                else:
                    samples = np.concatenate((samples, gen), axis=0)

            end = time.time()
            print("Generating samples took,", (end - start), "seconds")

            if last_batch != 0:
                if samples is None:
                    samples = self.sess.run(self.gen_out)[0:last_batch, ]
                else:
                    samples = np.concatenate((samples, self.sess.run(self.gen_out)[0:last_batch, ]), axis=0)

            end = time.time()
            print("Generating samples took,", (end - start), "seconds")

            if unique:
                start = time.time()
                samples = np.unique(samples, axis=0)
                end = time.time()
                print("Making samples unique took,", (end - start), "seconds")

            return samples
        else:
            return None

    def generate_single(self, n_samples, unique=True):
        if n_samples > 0:
            n_batches = int(n_samples // self.batch_size)

            start = time.time()
            i0 = tf.constant(0)
            m0 = self.gen_out

            c = lambda i, m: i < n_batches
            b = lambda i, m: [i + 1, tf.concat([m, self.gen_out], axis=0)]
            r = tf.while_loop(
                c, b, loop_vars=[i0, m0],
                shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])])

            _, samples = self.sess.run(r)
            end = time.time()
            print("Generating samples took,", (end - start), "seconds")

            if unique:
                start = time.time()
                samples = np.unique(samples, axis=0)
                end = time.time()
                print("Making samples unique took,", (end - start), "seconds")

            """ If number of samples is greater than n_samples """
            if samples.shape[0] > n_samples:
                discard = samples.shape[0] - n_samples
                indices = np.random.choice(samples.shape[0], size=discard, replace=False)
                samples = np.delete(samples, [indices], axis=0)

        return samples

    def prep(self, traces):
        input = []
        for trace in traces:
            t = []
            for ev in trace:
                t.append(int(self.wi_dict[ev]))
            while len(t) < self.seq_len:
                t.append(self.end_token)
            input.append(t)
        return np.array(input)

    def discriminate(self, data, threshold=None):
        n_batches = int(data.shape[0] // self.batch_size)
        last_batch = data.shape[0] % self.batch_size

        results = None
        for i in range(n_batches):
            dat = data[i * self.batch_size: (i + 1) * self.batch_size, ]
            if results is None:
                results = self.sess.run(self.disc_out, feed_dict={self.disc_in_x: dat})
            else:
                results = np.concatenate((results, self.sess.run(self.disc_out, feed_dict={self.disc_in_x: dat})),
                                         axis=0)

        if last_batch != 0:
            dat = data[(n_batches) * self.batch_size:, ]
            n_samples = dat.shape[0]

            #print(n_samples)

            missing = self.batch_size - dat.shape[0]
            dat = np.concatenate((dat, np.zeros((missing, dat.shape[1]), dtype=float)), axis=0)

            if results is None:
                results = self.sess.run(self.disc_out, feed_dict={self.disc_in_x: dat})[0:n_samples]
            else:
                results = np.concatenate(
                    (results, self.sess.run(self.disc_out, feed_dict={self.disc_in_x: dat})[0:n_samples]), axis=0)

        results = np.subtract(1, results)
        return results

    def to_one_hot(self, x):
        shape = x.shape  # batch_size x seqlen
        output = np.zeros(shape=[shape[0], shape[1], self.vocab_size])
        for row_index in range(shape[0]):
            for col_index in range(shape[1]):
                value = x[row_index, col_index]
                output[row_index, col_index, value] = 1
        return output
