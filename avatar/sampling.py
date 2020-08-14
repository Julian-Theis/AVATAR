import argparse
import os
import tensorflow as tf
import numpy as np

from avatar.util.LoadRelgan import LoadRelgan
from avatar.util.MHGAN import MHGAN
from avatar.util.util import writeToFile, readTraces

if __name__ == "__main__":
    np.random.seed(seed=1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', help='System to evaluate (e.g. pa_system_2_3)', required=True)
    parser.add_argument('-j', '--job', help='0 = beta 100, 1 = beta 1000', required=True)
    parser.add_argument('-sfx', '--suffix', help='Which suffix, i.e. final epoch of the trained SGAN to use?', required=True)
    parser.add_argument('-gpu', '--gpu', help='GPU on which the training is performed. For example 0.', required=True)

    """ Selector """
    parser.add_argument('-strategy', '--strategy', help='select "naive" or "mh"', required=True)

    """ Parameter for Naively Sampling """
    parser.add_argument('-n_n', '--n_samples', help='(NAIVE ONLY) Number of samples to generate? (Default: 10000)', default=10000)

    """ Parameter for MH Sampling """
    parser.add_argument('-mh_c', '--mh_count', help='(MH ONLY) Number of samples per batch? (Default: 50)',
                        default=50)
    parser.add_argument('-mh_p', '--mh_patience',
                        help='(MH ONLY) Patience parameter (Default: 5)',
                        default=5)
    parser.add_argument('-mh_k', '--mh_k',
                        help='(MH ONLY) Length of Markov chain (Default: 500)',
                        default=500)
    parser.add_argument('-mh_mi', '--mh_maxiter',
                        help='(MH ONLY) Max sampling iterations? (Default: 200)',
                        default=200)

    args = parser.parse_args()
    system = args.system
    job = int(args.job)
    suffix = args.suffix
    strategy = args.strategy
    n_samples = int(args.n_samples)
    mh_count = int(args.mh_count)
    mh_patience = int(args.mh_patience)
    mh_k = int(args.mh_count)
    mh_maxiter = int(args.mh_mi)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if strategy == "naive":
        tf.reset_default_graph()
        print("****** SAMPLE FOR SUFFIX ", suffix, " ******")
        relgan = LoadRelgan(system=system, suffix=suffix, job=job)

        f_out = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job) + "_naive.txt"
        print("Start NAIVE SAMPLING")
        gen_samples = relgan.generate(n_samples=n_samples)
        print("Generated samples - shape:", gen_samples.shape)
        print("Writing to file", f_out)
        writeToFile(relgan, f_out, gen_samples)

    elif strategy == "mh":
        eval_path = "data/avatar/train_data/" + system + "_eval.txt"
        f_out = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job) + "_mh.txt"

        tf.reset_default_graph()
        print("****** SAMPLE FOR SUFFIX ", suffix, " ******")
        relgan = LoadRelgan(system=system, suffix=suffix, job=job)

        calibrate = readTraces(eval_path)
        calibrate = relgan.prep(calibrate)

        mhgan = MHGAN(relgan, c=mh_count, k=mh_k, real_samples=calibrate)
        samples = None
        gen_size = 0
        iter = 1
        cnt_patience = 0
        continue_sampling = True

        print("Start MH SAMPLING")
        while continue_sampling:
            print("**** MH-GAN Iteration", iter, ":")
            gen_samples, accepts, rejects = mhgan.generate_enhanced(
                relgan.sess,
                count=mh_count,
                k=mh_k
            )

            if samples is None:
                samples = gen_samples
            else:
                samples = np.concatenate([samples, gen_samples], axis=0)

            samples = np.unique(samples, axis=0)

            if gen_size != samples.shape[0]:
                cnt_patience = 0
            else:
                cnt_patience += 1

            gen_size = samples.shape[0]
            print("Generated samples (cumulative): ", gen_size)
            iter += 1

            if cnt_patience >= mh_patience:
                continue_sampling = False

            if mh_maxiter != -1 and iter >= mh_maxiter:
                continue_sampling = False

        print("Generated samples - shape:", samples.shape)
        print("Writing to file", f_out)
        writeToFile(relgan, f_out, samples)

    else:
        raise ValueError("Unknown sampling strategy.")
