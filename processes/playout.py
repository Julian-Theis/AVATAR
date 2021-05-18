import os
import numpy as np
import argparse

from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.algo.simulation.playout.versions import basic_playout as playout
from util.playout import readVariantFile, getMaxVariantLength
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

def intersection(lst1, lst2):
    ls1 = []
    for i in lst1:
        ls1.append(str(i))

    ls2 = []
    for i in lst2:
        ls2.append(str(i))

    return list(set(ls1) & set(ls2))

def writeToFile(file, lst):
    with open(file, 'w') as outfile:
        traces = set()
        for entry in lst:
            trace = ""
            for index, ev in enumerate(entry):
                e = str(ev['concept:name'])
                e = e[0:-1]
                if index == 0:
                    trace = trace + str(e)
                else:
                    trace = trace + " " + str(e)
            traces.add(trace)

        for trace in traces:
            outfile.write(trace)
            outfile.write("\n")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', help='Which system? pb_system_5_3 for example', required=True)
    parser.add_argument('-e', '--eval_only', help='Evaluation only (Boolean)', required=True)
    parser.add_argument('-pn', '--pn', help='Petri net filename (for example fodina_pb_system_5_3_train.txt_0.4_1.0_0.4_true.pnml)', required=True)
    parser.add_argument('-traces', '--traces', help='Max number of traces to generate (default: 1,000,000)', default=1000000)
    args = parser.parse_args()

    system = args.system
    pn = args.pn
    n_traces = int(args.traces)
    eval_only = str2bool(args.eval_only)

    if DATA_PATH is None:
        f_pop = os.path.join(WORK_PATH, "data", "variants", str(system) + "_pop.txt")
        f_train = os.path.join(WORK_PATH, "data", "variants", str(system) + "_train.txt")
        f_test = os.path.join(WORK_PATH, "data", "variants", str(system) + "_test.txt")
        f_out = os.path.join(WORK_PATH, "data", "variants", pn + ".txt")
        f_pn = os.path.join(WORK_PATH, "data", "pns", system, pn)
    else:
        f_pop = os.path.join(DATA_PATH, "variants", str(system) + "_pop.txt")
        f_train = os.path.join(DATA_PATH, "variants", str(system) + "_train.txt")
        f_test = os.path.join(DATA_PATH, "variants", str(system) + "_test.txt")
        f_out = os.path.join(DATA_PATH, "variants", pn + ".txt")
        f_pn = os.path.join(DATA_PATH, "pns", system, pn)

    seq_len = getMaxVariantLength(f_pop)
    n_decimal = 8

    if eval_only:
        print("*** Variant Evaluation of " + system + " using " +  pn + " ***")
    else:
        print("*** Playout Variants of " + system + " and Evaluation using " +  pn + " ***")

    print("Maximum Variant Length is:", str(seq_len))

    if not eval_only:
        net, initial_marking, final_marking = pnml_importer.import_net(f_pn)
        out = playout.apply(net, initial_marking, parameters={"noTraces": n_traces, "maxTraceLength" : seq_len-1})
        writeToFile(f_out, out)

    train = readVariantFile(f_train, unique=True)
    test = readVariantFile(f_test, unique=True)
    pop = readVariantFile(f_pop, unique=True)
    gen = readVariantFile(f_out, unique=True)

    new_train = []
    for i in train:
        new_train.append([x.lower() for x in i])
    train = new_train

    new_test = []
    for i in test:
        new_test.append([x.lower() for x in i])
    test = new_test

    new_pop = []
    for i in pop:
        new_pop.append([x.lower() for x in i])
    pop = new_pop

    new_gen = []
    for i in gen:
        new_gen.append([x.lower() for x in i])
    gen = new_gen

    total_gen_samples = len(gen)
    cnt_true = 0
    labeled = []
    for sample in gen:
        label = 0
        if sample in pop:
            cnt_true = cnt_true + 1
            label = 1

        string = ""
        for i in sample:
            string = string + " " + i
        string = string + "," + str(label)
        labeled.append(string)

    print("** EVALUATION **")
    print("# System Variants:", len(pop))
    print("Approximated # System Variants:", total_gen_samples)
    print("TP:", np.round(cnt_true / total_gen_samples, n_decimal))
    print("TP_system:",
          np.round(len(intersection(gen, pop)) / len(pop), n_decimal))
    print("TP_observed:",
          np.round(len(intersection(gen, train)) / len(train), n_decimal))
    print("TP_unobserved:",
          np.round(len(intersection(gen, test)) / len(test), n_decimal))