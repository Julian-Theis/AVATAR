import numpy as np
import os
import argparse
from util.playout import readVariantFile
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

def readFile(f_name, unique=False):
    traces = []
    with open(f_name) as file:
        file_contents = file.read()
        file_contents = file_contents.split("\n")
        for row in file_contents:
            if unique:
                if row not in traces:
                    traces.append(row)
            else:
                traces.append(row)

    f_traces = []
    for trace in traces:
        f_trace = []
        t = trace.split(" ")
        for i in t:
            if i != "" and "<" not in i:
                f_trace.append(i)
        if len(f_trace) > 0:
            f_traces.append(f_trace)
    print(f_name, " length is :", str(len(f_traces)))
    return f_traces

def intersection(lst1, lst2):
    ls1 = []
    for i in lst1:
        ls1.append(str(i))

    ls2 = []
    for i in lst2:
        ls2.append(str(i))

    return list(set(ls1) & set(ls2))

def load_data(system, suffix, job, strategy=None):

    if DATA_PATH is None:
        f_train = os.path.join(WORK_PATH, "data", "variants", system + "_train.txt")
        f_test = os.path.join(WORK_PATH, "data", "variants", system + "_test.txt")
        f_pop = os.path.join(WORK_PATH, "data", "variants", system + "_pop.txt")
        f_eval = os.path.join(WORK_PATH, "data", "avatar", "train_data", system + "_eval.txt")
        if strategy is None:
            f_gan = os.path.join(WORK_PATH, "data", "avatar", "variants", system + "_relgan_" + str(suffix) + "_j" + str(job) + ".txt")
        else:
            f_gan = os.path.join(WORK_PATH, "data", "avatar", "variants", system + "_relgan_" + str(suffix) + "_j" + str(job) + "_" + strategy + ".txt")
    else:
        f_train = os.path.join(DATA_PATH, "variants", system + "_train.txt")
        f_test = os.path.join(DATA_PATH, "variants", system + "_test.txt")
        f_pop = os.path.join(DATA_PATH, "variants", system + "_pop.txt")
        f_eval = os.path.join(DATA_PATH, "avatar", "train_data", system + "_eval.txt")
        if strategy is None:
            f_gan = os.path.join(DATA_PATH, "avatar", "variants", system + "_relgan_" + str(suffix) + "_j" + str(job) + ".txt")
        else:
            f_gan = os.path.join(DATA_PATH, "avatar", "variants", system + "_relgan_" + str(suffix) + "_j" + str(job) + "_" + strategy + ".txt")


    train = readVariantFile(f_train, unique=False)
    test = readVariantFile(f_test, unique=False)
    eval = readVariantFile(f_eval, unique=False)
    pop = readVariantFile(f_pop, unique=False)
    gan = readVariantFile(f_gan, unique=True)

    return train, test, eval, pop, gan

def evaluate(suffix, train, test, eval, pop, gan):
    n_decimal = 6

    new_train = []
    for i in train:
        new_train.append([x.lower() for x in i])
    train = new_train

    new_eval = []
    for i in eval:
        new_eval.append([x.lower() for x in i])
    eval = new_eval

    new_test = []
    for i in test:
        new_test.append([x.lower() for x in i])
    test = new_test

    new_pop = []
    for i in pop:
        new_pop.append([x.lower() for x in i])
    pop = new_pop

    new_gan = []
    for i in gan:
        new_gan.append([x.lower() for x in i])
    gan = new_gan

    total_gan_samples = len(gan)
    cnt_true = 0

    labeled = []
    for sample in gan:
        label = 0
        if sample in pop:
            cnt_true = cnt_true + 1
            label = 1

        string = ""
        for i in sample:
            string = string + " " + i
        string = string + "," + str(label)
        labeled.append(string)

    tp = np.round(cnt_true / total_gan_samples, n_decimal)
    tp_pop = np.round(len(intersection(gan, pop)) / len(pop), n_decimal)
    tp_train = np.round(len(intersection(gan, train)) / len(train), n_decimal)
    tp_eval = np.round(len(intersection(gan, eval)) / len(eval), n_decimal)
    tp_test = np.round(len(intersection(gan, test)) / len(test), n_decimal)

    print("***** EVALUATION of Suffix ", suffix, "*****")
    print("# of System Variants:", len(pop))
    print("Approximated # of System Variants:", total_gan_samples)
    print("TP Ratio of realistic GAN samples over all GAN samples (TP):", tp)
    print("TP system:", tp_pop)
    print("TP observed:", tp_train)
    print("TP unobserved", tp_test)
    print("TP eval:", tp_eval)

    dict_suffix = dict()
    dict_suffix["sampled"] = total_gan_samples
    dict_suffix["tp"] = tp
    dict_suffix["tp_pop"] = tp_pop
    dict_suffix["tp_train"] = tp_train
    dict_suffix["tp_test"] = tp_test
    dict_suffix["tp_eval"] = tp_eval

    return dict_suffix

def eval_sgan_single(system, suffix, job):
    train, test, eval, pop, gan = load_data(system, suffix, job)
    eval(suffix, train, test, eval, pop, gan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sfx', '--suffix', help='Suffix (selected epoch, e.g. 1981)', required=True)
    parser.add_argument('-s', '--system', help='System name', required=True)
    parser.add_argument('-j', '--job', help='Job (0,1)', required=True)
    parser.add_argument('-strategy', '--strategy', help='naive/mh', required=True)

    args = parser.parse_args()

    system = args.system
    suffix = int(args.suffix)
    job = int(args.job)
    strategy = args.strategy

    train, test, eval, pop, gan = load_data(system, suffix, job, strategy=strategy)
    evaluate(suffix, train, test, eval, pop, gan)
