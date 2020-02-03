import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

class MHGAN:
    '''
    Wraps your trained WGAN with generator and discriminator for enhanced
    output sampling.
    '''
    def __init__(self, relgan, c, k, real_samples):
        self.relgan = relgan
        self.real_samples = real_samples
        self.generator_output_shape = relgan.gen_out.shape
        self.generator_output_tensor = relgan.gen_out

        self.discriminator_input_tensor = relgan.disc_in_x
        self.discriminator_output_tensor = relgan.disc_out

        with tf.name_scope('MH'):
            self.c = tf.placeholder(tf.int32, [], name='total_count')
            self.k = tf.placeholder(tf.int32, [], name='k_count')

            self.real_output_tensor = tf.placeholder(tf.float32, name='real_output_tensor', shape=(c, ))
            self.fake_output_tensor = tf.placeholder(tf.float32, name='fake_output_tensor', shape=((c * k), ))

            self.u = tf.random_uniform([self.c, self.k ])

            # Scores for calibration + generated samples
            # self.scores = tf.concat([tf.sigmoid(self.real_output_tensor), tf.sigmoid(self.fake_output_tensor)], 0)

            self.scores = tf.reshape(
                # Add calibration scores from real discriminator output
                #tf.concat([tf.sigmoid(self.real_output_tensor), tf.sigmoid(self.fake_output_tensor)], 0),
                tf.concat([self.real_output_tensor, self.fake_output_tensor], 0),
                (self.c, self.k + 1)
            )

    def generate(self, sess, count=1, squeeze=True):
        '''
        Draws <count> number of samples from Generator
        '''
        #samples = sess.run(self.generator_output_tensor)
        samples = self.generate_raw(sess=sess, n_samples=count)
        if squeeze:
            return samples.squeeze(axis=3)
        return samples

    def generate_enhanced(self, sess, count=1, k=100, squeeze=True):
        '''
        Draws <count> number of enhanced samples from Generator with
        Metropolis-Hastings algorithm.
        '''
        real_samples = self.sample_from_samples(count=count, data=self.real_samples)
        fake_samples = self.generate(sess, count=(count * k), squeeze=False)
        samples = np.concatenate([real_samples, fake_samples])

        disc_real_samples = self.relgan.discriminate(real_samples, threshold=None)
        disc_fake_samples = self.relgan.discriminate(fake_samples, threshold=None)

        epsilon = sess.run(self.u, feed_dict={self.c: count,
                                              self.k: k
                                             })

        scores = sess.run(self.scores, feed_dict={self.real_output_tensor : disc_real_samples,
                                                  self.fake_output_tensor : disc_fake_samples,
                                                  self.c: count,
                                                  self.k: k})

        selected, accepted, rejected = self.metropolishastings(count=count, k=k, epsilon=epsilon, samples=samples, scores=scores)
        return selected, accepted, rejected


    def generate_enhanced_pn(self, sess, gen_pn, count=1, k=100, squeeze=True):
        '''
        Draws <count> number of enhanced samples from Generator with
        Metropolis-Hastings algorithm.
        '''
        print("Real samples", self.real_samples.shape)
        real_samples = self.sample_from_samples(count=count, data=self.real_samples)
        print("Real samples sampled", real_samples.shape)

        fake_samples = self.sample_from_samples(count=(count * k), data=gen_pn)
        samples = np.concatenate([real_samples, fake_samples])


        disc_real_samples = self.relgan.discriminate(real_samples, threshold=None)
        print("Discriminated real samples", disc_real_samples.shape)

        disc_fake_samples = self.relgan.discriminate(fake_samples, threshold=None)
        print("Discriminated fake samples", disc_fake_samples.shape)

        #print("Discriminated REAL samples")
        avg_d_real = np.mean(disc_real_samples)
        print(avg_d_real)

        plt.hist(disc_real_samples, color='blue', edgecolor='black',
                 bins=int(180 / 5))
        plt.savefig("/data/julian/data/ganeval/plots/real.png")
        plt.close()

        #print("Discriminated FAKE samples")
        print(np.mean(disc_fake_samples))

        plt.hist(disc_fake_samples, color='blue', edgecolor='black',
                 bins=int(180 / 5))
        plt.savefig("/data/julian/data/ganeval/plots/test.png")
        plt.close()


        epsilon = sess.run(self.u, feed_dict={self.c: count,
                                              self.k: k
                                              })

        #print("Real output tensor shape:", self.real_output_tensor.shape)
        #print("Fake output tensor shape:", self.fake_output_tensor.shape)
        #print("Real samples discriminated:", disc_real_samples.shape)
        #print("Fake samples discriminated:", disc_fake_samples.shape)

        scores = sess.run(self.scores, feed_dict={self.real_output_tensor: disc_real_samples,
                                                  self.fake_output_tensor: disc_fake_samples,
                                                  self.c: count,
                                                  self.k: k})

        #print("Scores shape:", scores.shape)

        selected, accepted, rejected = self.metropolishastings(count=count, k=k, epsilon=epsilon, samples=samples,
                                                               scores=scores)
        return selected, accepted, rejected


    def metropolishastings(self, count, k, scores, epsilon, samples, squeeze=True):
        # Metropolis-Hastings GAN algorithm
        selected = []
        accepts = 0
        rejects = 0
        for i in range(count):
            x = 0
            for x_next in range(k):
                Pd1 = scores[i][x]
                Pd2 = scores[i][x_next]
                alpha = np.fmin(1., np.true_divide((1./Pd1 - 1.), (1./Pd2 - 1.)))
                # Will ignore NaNs
                if epsilon[i][x_next] <= alpha:
                    x = x_next
                    accepts = accepts + 1
                else:
                    rejects = rejects + 1
                # Avoid samples from calibration distribution
                x += int(x == 0)
            #print("Append", x)
            selected.append(samples[x])
        selected = np.asarray(selected)
        if squeeze and selected.ndim > 3:
            return selected.squeeze(axis=3)

        #print("TOTAL ACCEPTS:", accepts)
        #print("TOTAL REJECTS:", rejects)
        #print("ACCEPTANCE RATIO IS:", float(accepts) / float((accepts + rejects)))
        return selected, accepts, rejects



    """ JT OWN FUNCTIONS """
    def generate_raw(self, sess, n_samples, unique=True):
        samples = None
        sample_size = 0
        while sample_size < n_samples:
            if samples is None:
                samples = sess.run(self.generator_output_tensor)
                if unique:
                    samples = np.unique(samples, axis=0)
            else:
                s = sess.run(self.generator_output_tensor)
                if unique:
                    s = np.unique(s, axis=0)
                samples = np.concatenate((samples, s), axis=0)
            sample_size = samples.shape[0]

        if sample_size > n_samples:
            samples = samples[0:n_samples, :]

        return samples

    def sample_from_samples(self, count, data):
        subset_size = len(data)
        sampled = None
        sample_size = 0
        while sample_size < count:
            if sampled is None:
                sampled = data[:subset_size][np.random.permutation(subset_size)][:count]
            else:
                sampled = np.concatenate((sampled, data[:subset_size][np.random.permutation(subset_size)][:count]), axis=0)
            sample_size = sampled.shape[0]

        if sample_size > count:
            sampled = sampled[0:count, :]
        return sampled