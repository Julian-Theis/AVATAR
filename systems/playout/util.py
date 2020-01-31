import os
import numpy as np
from datetime import datetime

from pm4py.objects.petri.importer import factory as pnml_importer
from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from sklearn.model_selection import train_test_split

from processmining.playout import Player

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

    writeVariantToFile(f_train, train)
    print("Unique Variant Log (Train, " + str(int(train_size*100)) + "%) stored in:", str(f_train))

    writeVariantToFile(f_test, test)
    print("Unique Variant Log (Test, "+ str(int((1-train_size)*100)) +"%) stored in:", str(f_test))

    prepare_xes_csv(f_train, xes_train, csv_train)
    printStatistics(f_pop, f_train, f_test)


def biased_playout(pn, f_pop, f_train, f_test, xes_train, csv_train,):
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

    writeVariantToFile(f_train, train)
    print("Biased Unique Variant Log (Train, 70%) stored in:", str(f_train))

    writeVariantToFile(f_test, test)
    print("Biased Unique Variant Log (Test, 30%) stored in:", str(f_test))

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