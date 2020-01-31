import os
import tensorflow as tf
import argparse
import numpy as np
import json
from sklearn.model_selection import train_test_split
from util.playout import readVariantFile, writeVariantToFile
from avatar.relgan.run import main as relgan_main
from avatar.util.LoadRelgan import LoadRelgan
from avatar.evaluation import load_data, evaluate
from avatar.util.util import writeToFile

def find_num_sentences(system):
    file = "data/avatar/train_data/" + system + ".txt"
    return str(sum(1 for _ in open(file)))

def split_train_eval(system, ratio=0.1):
    print("*** Splitting System", system, " to train and evaluation with ratio ", str(ratio), "***")

    load_file = "data/variants/" + str(system) + "_train.txt"
    train_out = "data/avatar/train_data/" + str(system) + ".txt"
    eval_out = "data/avatar/train_data/" + str(system) + "_eval.txt"

    all_sequences = readVariantFile(load_file, unique=False)
    indices = np.arange(0, len(all_sequences))
    np.random.shuffle(indices)

    train, eval, _, _ = train_test_split(indices, indices, test_size=ratio, random_state=42)

    train_data = []
    eval_data = []
    for index in train:
        train_data.append(all_sequences[index])
    for index in eval:
        eval_data.append(all_sequences[index])

    writeVariantToFile(train_out, train_data)
    writeVariantToFile(eval_out, eval_data)
    print("*** Train and eval files stored. ***")

if __name__ == "__main__":
    np.random.seed(seed=1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', help='System to evaluate (e.g. pa_system_2_3)', required=True)
    parser.add_argument('-j', '--job', help='0 = beta 100, 1 = beta 1000', required=True)
    parser.add_argument('-gpu', '--gpu', help='GPU on which the training is performed. For example 0.', required=True)
    parser.add_argument('-n', '--n_samples', help='Number of samples to evaluate the best model', default=10000)
    args = parser.parse_args()

    system = args.system
    job_id = int(args.job)
    gpu_id = args.gpu
    n_samples = int(args.n_samples)

    split_train_eval(system, ratio=0.1)


    if system == 'pa_system_4_3':
        bs = '32'
    else:
        bs = '64'

    # Executables
    executable = 'python3'

    architecture = 'rmc_vanilla'
    gantype = 'RSGAN'
    opt_type = 'adam'
    seed = '172'
    num_heads = '2'
    head_size = '256'
    mem_slots = '1'
    d_lr = '1e-4'
    gadv_lr = '1e-4'

    num_sentences = find_num_sentences(system)

    # Arguments
    temperature = ['100', '1000']

    gpre_lr = '1e-2'
    hidden_dim = '16'

    gsteps = '1'
    dsteps = '5'
    gen_emb_dim = '32'
    dis_emb_dim = '64'
    num_rep = '64'
    sn = False
    decay = False
    adapt = 'exp'
    npre_epochs = '100' #100
    nadv_steps = '5000' #5000
    ntest = '20'

    # Paths
    scriptname = 'run.py'
    cwd = os.path.dirname(os.path.abspath(__file__))
    rootdir = cwd + "/.."

    outdir = os.path.join("data/avatar/sgans/", system, str(job_id))

    args = [
        # Architecture
        '--gf-dim', '64',
        '--df-dim', '64',
        '--g-architecture', architecture,
        '--d-architecture', architecture,
        '--gan-type', gantype,
        '--hidden-dim', hidden_dim,

        # Training
        '--gsteps', gsteps,
        '--dsteps', dsteps,
        '--npre-epochs', npre_epochs,
        '--nadv-steps', nadv_steps,
        '--ntest', ntest,
        '--d-lr', d_lr,
        '--gpre-lr', gpre_lr,
        '--gadv-lr', gadv_lr,
        '--batch-size', bs,
        '--log-dir', os.path.join(outdir, 'tf_logs'),
        '--sample-dir', os.path.join(outdir, 'samples'),
        '--optimizer', opt_type,
        '--seed', seed,
        '--temperature', temperature[job_id],
        '--adapt', adapt,

        # evaluation
        '--nll-gen',
        '--bleu',

        # relational memory
        '--mem-slots', mem_slots,
        '--head-size', head_size,
        '--num-heads', num_heads,

        # dataset
        '--dataset', system,
        '--start-token', '0',
        '--num-sentences', num_sentences,  # how many generated sentences to use per evaluation
        '--gen-emb-dim', gen_emb_dim,
        '--dis-emb-dim', dis_emb_dim,
        '--num-rep', num_rep,
    ]

    if sn:
        args += ['--sn']
    if decay:
        args += ['--decay']

    # Run
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    """ Train RelGAN """
    relgan_main(given_args=args)


    """ Sample from all """
    ranges = range(1, int(nadv_steps), int(ntest))

    for suffix in ranges:
        tf.reset_default_graph()
        suffix = str(suffix)
        print("****** SAMPLE FOR SUFFIX ", suffix, " ******")
        relgan = LoadRelgan(system=system, suffix=suffix, job=job_id)

        f_out = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job_id) + ".txt"
        print("Start sampling")
        gen_samples = relgan.generate(n_samples=n_samples)
        print(gen_samples.shape)
        print("Done sampling")
        print("Writing to file")
        writeToFile(relgan, f_out, gen_samples)

    """ Evaluate All """
    train, test, eval, pop, _ = load_data(system, "1", job_id)
    results = dict()
    for suffix in ranges:
        f_gan = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job_id) + ".txt"
        gan = readVariantFile(f_gan, unique=True)
        res = evaluate(suffix, train, test, eval, pop, gan)
        results[suffix] = res

    iw = json.dumps(results)
    f = open(os.path.join("data/avatar/sgans", system, str(job_id), "evaluations_relgan_" + str(system) + "_" + str(job_id) + ".json"), "w")
    f.write(iw)
    f.close()

    """ Show ten best models """
    print("")
    print("")
    print("*** FIND BEST SUFFIX ***")
    xs, sampled, tps, tps_pop, tps_train, tps_test, tps_eval = list(), list(), list(), list(), list(), list(), list()
    for i in results.keys():
        xs.append(int(i))
        sampled.append(int(results[i]["sampled"]))
        tps.append(float(results[i]["tp"]))
        tps_pop.append(float(results[i]["tp_pop"]))
        tps_train.append(float(results[i]["tp_train"]))
        tps_test.append(float(results[i]["tp_test"]))
        tps_eval.append(float(results[i]["tp_eval"]))

    top_10_idx = np.argsort(tps_eval)[-10:]
    top_10_values = [tps_eval[i] for i in top_10_idx]

    ranks = dict()
    for i in top_10_idx:
        obj = dict()
        print("*** RANK " + str(i+1) + " ****")
        print("Suffix: ", xs[i])
        print("TP observed: ", tps_train[i])
        print("TP eval: ", tps_eval[i])
        print("")
        obj["suffix"] = xs[i]
        obj["tp_obs"] = tps_train[i]
        obj["tp_eval"] = tps_eval[i]
        ranks[str(i+1)] = obj

    iw = json.dumps(ranks)
    f = open(os.path.join("data/avatar/sgans", system, str(job_id),
                          "suffix_ranks_relgan_" + str(system) + "_" + str(job_id) + ".json"), "w")
    f.write(iw)
    f.close()
    print("Ranks saved to file.")