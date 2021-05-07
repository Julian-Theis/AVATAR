import os
import random
import numpy as np

from queue import Queue
from copy import copy
from datetime import datetime

from pm4py.objects.petri import semantics
from pm4py.objects.petri.importer import factory as pnml_importer
from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from sklearn.model_selection import train_test_split

class PotentialTrace():
    def __init__(self, marking, firingSequence, marking_count, inv_counter=0):
        self.marking = marking
        self.firingSequence = firingSequence
        self.inv_counter = inv_counter
        self.marking_count = marking_count

    def getMarking(self):
        return self.marking

    def getMarkingCount(self):
        return self.marking_count

    def getFiringSequence(self):
        return self.firingSequence

    def getInvCounter(self):
        return self.inv_counter

class Player():
    def __init__(self, net, initial_marking, final_marking, maxTraceLength, rep_inv_thresh=100, max_loop=3):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.maxTraceLength = maxTraceLength
        self.rep_inv_thresh = rep_inv_thresh
        self.max_loop = max_loop

    def play(self):
        self.generatedTraces = set()
        self.potentials = Queue()

        marking_count = dict()
        for mark in self.initial_marking:
            marking_count[mark] = 1

        self.potentials.put_nowait(PotentialTrace(marking = copy(self.initial_marking), firingSequence=list(), marking_count=marking_count))

        while not self.potentials.empty():
            potential = self.potentials.get_nowait()
            marking = potential.getMarking()
            firingSeq = potential.getFiringSequence()

            enabled_trans = semantics.enabled_transitions(self.net, marking)
            for enabled_tran in enabled_trans:
                new_marking = semantics.execute(enabled_tran, self.net, marking)
                new_firingSeq = copy(firingSeq)

                discard = False
                marking_count = copy(potential.getMarkingCount())
                for mark in new_marking:
                    if mark not in marking:
                        if mark in marking_count.keys():
                            marking_count[mark] = marking_count[mark] + 1
                            if marking_count[mark] > self.max_loop:
                                discard = True
                        else:
                            marking_count[mark] = 1

                if enabled_tran.label == None:
                    invs = potential.getInvCounter() + 1
                else:
                    new_firingSeq.append(str(enabled_tran))
                    invs = 0

                if new_marking == self.final_marking:
                    self.generatedTraces.add(tuple(new_firingSeq))
                else:
                    if len(new_firingSeq) < self.maxTraceLength and invs < self.rep_inv_thresh and not discard:
                        self.potentials.put_nowait(PotentialTrace(marking=new_marking, firingSequence=new_firingSeq, inv_counter=invs, marking_count=marking_count))
        return self.generatedTraces


class Sampler():
    def __init__(self, net, initial_marking, final_marking, samples=100):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.samples = samples

    def play(self):
        self.generatedTraces = set()
        for sample in range(self.samples):
            firingSeq = list()
            marking = self.initial_marking
            while marking != self.final_marking:
                enabled_trans = semantics.enabled_transitions(self.net, marking)
                if len(enabled_trans) > 0:
                    tran = random.choice([i for i in enabled_trans])

                    if tran.label != None:
                        firingSeq.append(str(tran))
                    marking = semantics.execute(tran, self.net, marking)
                    self.generatedTraces.add(tuple(firingSeq))
                else:
                    firingSeq = list()
                    marking = self.initial_marking
        return self.generatedTraces

def standard_playout(pn, f_pop, f_train, f_test, xes_train, csv_train, train_size=0.7):
    print("*** PLAYOUT " + str(pn) + " ***")
    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print("Initial Marking:", initial_marking)
    print("Final Marking:",final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()

    print("Total Number of Variants:", len(gen_traces))
    writeVariantToFile(f_pop, gen_traces)
    print("Unique Variant Log stored in:", str(f_pop))

    max_len = 0
    for trace in gen_traces:
        if len(trace) > max_len:
            max_len = len(trace)

    test = list()
    train = list()
    for trace in gen_traces:
        if len(trace) == max_len:
            train.append(trace)
            gen_traces.remove(trace)
            break

    indices = [i for i in range(0, len(gen_traces))]
    train_indices, test_indices = train_test_split(indices, train_size=0.7)

    for index, trace in enumerate(gen_traces):
        if index in train_indices:
            train.append(trace)
        else:
            test.append(trace)

    print()
    print("Mean variant length size train:", np.mean([len(s) for s in train]))
    print("Mean variant length size test:", np.mean([len(s) for s in test]))
    print()
    print("Max variant length size train:", np.max([len(s) for s in train]))
    print("Max variant length size test:", np.max([len(s) for s in test]))
    print()

    writeVariantToFile(f_train, train)
    print("Unique Variant Log (Train, " + str(int(train_size*100)) + "%) stored in:", str(f_train))

    writeVariantToFile(f_test, test)
    print("Unique Variant Log (Test, "+ str(int((1-train_size)*100)) +"%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)


def biased_playout_b1(pn, f_pop, f_train, f_test, xes_train, csv_train,):
    print("*** PLAYOUT " + str(pn) + " ***")
    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print("Initial Marking:", initial_marking)
    print("Final Marking:",final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()

    print("Total Number of Variants:", len(gen_traces))
    writeVariantToFile(f_pop, gen_traces)
    print("Unique Variant Log stored in:", str(f_pop))

    max_len = 0
    for trace in gen_traces:
        if len(trace) > max_len:
            max_len = len(trace)

    test = list()
    train = list()
    for trace in gen_traces:
        if len(trace) == max_len:
            train.append(trace)
            gen_traces.remove(trace)
            break

    gen_traces = list(gen_traces)
    gen_traces.sort(key=lambda s: len(s))
    threshold = 0.7 * len(gen_traces)

    cnt = 0
    for trace in gen_traces:
        if cnt < threshold:
            train.append(trace)
        else:
            test.append(trace)
        cnt += 1

    print()
    print("Mean variant length size train:", np.mean([len(s) for s in train]))
    print("Mean variant length size test:", np.mean([len(s) for s in test]))
    print()
    print("Max variant length size train:", np.max([len(s) for s in train]))
    print("Max variant length size test:", np.max([len(s) for s in test]))
    print()

    writeVariantToFile(f_train, train)
    print("b1 Biased Unique Variant Log (Train, 70%) stored in:", str(f_train))

    writeVariantToFile(f_test, test)
    print("b1 Biased Unique Variant Log (Test, 30%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)

def biased_playout_b2(pn, f_pop, f_train, f_test, xes_train, csv_train,):
    print("*** PLAYOUT " + str(pn) + " ***")
    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print("Initial Marking:", initial_marking)
    print("Final Marking:",final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()

    print("Total Number of Variants:", len(gen_traces))
    writeVariantToFile(f_pop, gen_traces)
    print("Unique Variant Log stored in:", str(f_pop))

    max_len = 0
    for trace in gen_traces:
        if len(trace) > max_len:
            max_len = len(trace)

    test = list()
    train = list()

    gen_traces = list(gen_traces)
    gen_traces.sort(key=lambda s: -len(s))

    threshold = 0.7 * len(gen_traces)

    cnt = 0
    for trace in gen_traces:
        if cnt < threshold:
            train.append(trace)
        else:
            test.append(trace)
        cnt += 1

    print()
    print("Mean variant length size train:", np.mean([len(s) for s in train]))
    print("Mean variant length size test:", np.mean([len(s) for s in test]))
    print()
    print("Max variant length size train:", np.max([len(s) for s in train]))
    print("Max variant length size test:", np.max([len(s) for s in test]))
    print()

    writeVariantToFile(f_train, train)
    print("b2 Biased Unique Variant Log (Train, 70%) stored in:", str(f_train))

    writeVariantToFile(f_test, test)
    print("b2 Biased Unique Variant Log (Test, 30%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)


def biased_playout_b3(pn, f_pop, f_train, f_test, xes_train, csv_train,):
    print("*** PLAYOUT " + str(pn) + " ***")
    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print("Initial Marking:", initial_marking)
    print("Final Marking:",final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()

    print("Total Number of Variants:", len(gen_traces))
    writeVariantToFile(f_pop, gen_traces)
    print("Unique Variant Log stored in:", str(f_pop))

    max_len = 0
    for trace in gen_traces:
        if len(trace) > max_len:
            max_len = len(trace)

    test = list()
    train = list()
    for trace in gen_traces:
        if len(trace) == max_len:
            train.append(trace)
            gen_traces.remove(trace)
            break

    gen_traces = list(gen_traces)
    gen_traces.sort(key=lambda s: len(s))
    threshold = 0.7 * len(gen_traces)

    cnt = 0
    for trace in gen_traces:
        if cnt < threshold:
            train.append(trace)
        else:
            test.append(trace)
        cnt += 1

    r_train = []
    r_test = []
    p_thresh = 0.75

    for t in test:
        p = np.random.uniform(0, 1, 1)[0]
        if p < p_thresh:
            r_test.append(t)
        else:
            r_train.append(t)

    required_transfers = len(r_train)
    train_transfer = random.sample(train, required_transfers)
    while np.max([len(s) for s in train_transfer]) == max_len:
        train_transfer = random.sample(train, required_transfers)
        print("resample")

    for t in train:
        if t in train_transfer:
            r_test.append(t)
        else:
            r_train.append(t)

    print()
    print("Transfer", len(train_transfer), "variants from train to test.")
    print("Mean variant length size train (before randomization):", np.mean([len(s) for s in train]))
    print("Mean variant length size test (before randomization):", np.mean([len(s) for s in test]))
    print("Mean variant length size train (after randomization):", np.mean([len(s) for s in r_train]))
    print("Mean variant length size test (after randomization):", np.mean([len(s) for s in r_test]))
    print("Size before and after randomization (train):", len(train), len(r_train))
    print("Size before and after randomization (test):",len(test), len(r_test))
    print()
    print("Max variant length size train (after randomization):", np.max([len(s) for s in r_train]))
    print("Max variant length size test (after randomization):", np.max([len(s) for s in r_test]))
    print()
    writeVariantToFile(f_train, r_train)
    print("b3 Biased Unique Variant Log (Train, 70%) stored in:", str(f_train))

    writeVariantToFile(f_test, r_test)
    print("b3 Biased Unique Variant Log (Test, 30%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)

def biased_playout_b4(pn, f_pop, f_train, f_test, xes_train, csv_train,):
    print("*** PLAYOUT " + str(pn) + " ***")
    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print("Initial Marking:", initial_marking)
    print("Final Marking:",final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()

    print("Total Number of Variants:", len(gen_traces))
    writeVariantToFile(f_pop, gen_traces)
    print("Unique Variant Log stored in:", str(f_pop))

    max_len = 0
    for trace in gen_traces:
        if len(trace) > max_len:
            max_len = len(trace)

    test = list()
    train = list()

    gen_traces = list(gen_traces)
    gen_traces.sort(key=lambda s: -len(s))

    threshold = 0.7 * len(gen_traces)

    cnt = 0
    for trace in gen_traces:
        if cnt < threshold:
            train.append(trace)
        else:
            test.append(trace)
        cnt += 1

    r_train = []
    r_test = []
    p_thresh = 0.75

    for t in test:
        p = np.random.uniform(0, 1, 1)[0]
        if p < p_thresh:
            r_test.append(t)
        else:
            r_train.append(t)

    required_transfers = len(r_train)
    train_transfer = random.sample(train, required_transfers)
    while np.max([len(s) for s in train_transfer]) == max_len:
        train_transfer = random.sample(train, required_transfers)
        print("resample")

    for t in train:
        if t in train_transfer:
            r_test.append(t)
        else:
            r_train.append(t)

    print()
    print("Transfer", len(train_transfer), "variants from train to test.")
    print("Mean variant length size train (before randomization):", np.mean([len(s) for s in train]))
    print("Mean variant length size test (before randomization):", np.mean([len(s) for s in test]))
    print("Mean variant length size train (after randomization):", np.mean([len(s) for s in r_train]))
    print("Mean variant length size test (after randomization):", np.mean([len(s) for s in r_test]))
    print("Size before and after randomization (train):", len(train), len(r_train))
    print("Size before and after randomization (test):",len(test), len(r_test))
    print()
    print("Max variant length size train (after randomization):", np.max([len(s) for s in r_train]))
    print("Max variant length size test (after randomization):", np.max([len(s) for s in r_test]))
    print()
    writeVariantToFile(f_train, r_train)
    print("b4 Biased Unique Variant Log (Train, 70%) stored in:", str(f_train))

    writeVariantToFile(f_test, r_test)
    print("b4 Biased Unique Variant Log (Test, 30%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)


def prepare_xes_csv(f_train, xes_train, csv_train):
    train = readVariantFile(f_train, unique=False)

    convertToCsv(train, csv_train)
    print("CSV Event Log stored in:", str(csv_train))

    train_log = csv_importer.import_event_log(csv_train)
    xes_exporter.export_log(train_log, xes_train)
    print("XES Event Log stored in:", str(xes_train))

def writeVariantToFile(file, lst):
    with open(file, 'w') as outfile:
        for entry in lst:
            print_trace = ""
            for index, ev in enumerate(entry):
                if index == 0:
                    print_trace = str(ev).replace(" ", "")
                else:
                    print_trace = print_trace + " " + str(ev).replace(" ", "")
            outfile.write(print_trace.strip() + "\n")


def readVariantFile(f_name, unique=False):
    """
    Reads Variant File

    :param f_name: filename
    :param unique: True or False
    :return:
    """
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
                f_trace.append(str(i))
        if len(f_trace) > 0:
            f_traces.append(f_trace)

    return f_traces

def writeLinesToFile(file, lst):
    with open(file, 'w') as outfile:
        for entry in lst:
            outfile.write(str(entry) + "\n")


def convertToCsv(file, out_file):
    lines = []

    case = 0
    timestamp = 0
    line = "concept:name,case:concept:name,time:timestamp"
    lines.append(line)
    for trace in file:
        for event in trace:
            timestamp = timestamp + 1
            dt_object = datetime.fromtimestamp(timestamp)

            line = str(event) + "_" + "," + str(case) + "," + str(dt_object)
            lines.append(line)

        case = case + 1

    filename = os.path.join(out_file)
    writeLinesToFile(filename, lines)

def printStatistics(f_pop, f_train, f_test):
    train = readVariantFile(f_train)
    test = readVariantFile(f_test)
    pop = readVariantFile(f_pop)

    length = max(map(len, pop))
    unique = np.array([xi + [None] * (length - len(xi)) for xi in pop])
    unique = unique.flatten()
    unique = np.array(list(filter(None, unique)))
    unique = np.unique(unique)
    print("*** System Details ***")
    print("# unique events:", len(unique))
    print("# train variants:", len(train))
    print("# test variants:", len(test))
    print("# system variants:", len(pop))

    max_len = 0
    for trace in pop:
        if len(trace) > max_len:
            max_len = len(trace)

    print("Max system variant length:", max_len)

def getMaxVariantLength(f_pop):
    pop = readVariantFile(f_pop)
    max_len = 0
    for trace in pop:
        if len(trace) > max_len:
            max_len = len(trace)
    return max_len