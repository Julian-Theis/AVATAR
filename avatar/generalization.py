import os, time, argparse
from datetime import datetime

from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.petri.importer import pnml as pnml_importer

from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.evaluation.precision import factory as precision_factory

def readFile(f_name1, f_name2, unique=False):
    traces = []

    skipped = 0

    with open(f_name1) as file:
        file_contents = file.read()
        file_contents = file_contents.split("\n")
        print("Number of train traces are:", str(len(file_contents)))
        for row in file_contents:
            if unique:
                if row not in traces:
                    traces.append(row)
                else:
                    skipped += 1
            else:
                traces.append(row)

    with open(f_name2) as file:
        file_contents = file.read()
        file_contents = file_contents.split("\n")
        print("Number of generated traces are:", str(len(file_contents)))
        for row in file_contents:
            if unique:
                if row not in traces:
                    traces.append(row)
                else:
                    skipped += 1
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

    print("Number of traces are:", str(len(f_traces)))
    print("Number of skipped traces are:", str(skipped))
    return f_traces

def writeToFile(file, lst):
    with open(file, 'w') as outfile:
        for entry in lst:
            outfile.write(str(entry) + "\n")

def convertToCsv(traces, to_path):
    lines = []

    case = 0
    timestamp = 0
    line = "concept:name,case:concept:name,time:timestamp"
    lines.append(line)
    for trace in traces:
        for event in trace:
            timestamp = timestamp + 1
            dt_object = datetime.fromtimestamp(timestamp)

            line = str(event) + "_" + "," + str(case) + "," + str(dt_object)
            lines.append(line)

        case = case + 1

    writeToFile(str(to_path), lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', help='Which system (e.g. pb_system_5_3)', required=True)
    parser.add_argument('-sfx', '--suffix', help='Suffix (chosen epoch, e.g. 1981)', required=True)
    parser.add_argument('-j', '--job', help='Job (0/1)', required=True)
    parser.add_argument('-pn', '--pn', help='Petri net file to evaluate', required=True)
    parser.add_argument('-strategy', '--strategy', help='naive/mh', required=True)
    args = parser.parse_args()

    system = args.system
    suffix = int(args.suffix)
    job = args.job
    pn = args.pn
    strategy = args.strategy

    train_file = "data/variants/" + system + "_train.txt"
    gen_file = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job) + "_" + strategy + ".txt"
    csv_file = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job) + "_" + strategy + "_generalization.csv"
    xes_file = "data/avatar/variants/" + system + "_relgan_" + str(suffix) + "_j" + str(job) + "_" + strategy + "_generalization.xes"

    pn_file = os.path.join('data/pns', system, pn)

    """ READ FILES AND CONVERT TO XES """
    traces = readFile(train_file,gen_file, unique=True)
    convertToCsv(traces=traces, to_path=csv_file)
    time.sleep(1)

    log = csv_importer.import_event_log(csv_file)
    xes_exporter.export_log(log, xes_file)
    time.sleep(1)

    """ PERFORM MEASUREMENT ON PN AND XES"""
    log = xes_importer.import_log(xes_file)
    net, initial_marking, final_marking = pnml_importer.import_net(pn_file)

    fitness = replay_factory.apply(log, net, initial_marking, final_marking)
    print("Fitness=", fitness)

    precision = precision_factory.apply(log, net, initial_marking, final_marking)
    print("Precision=", precision)

    fitness = fitness["log_fitness"]
    generalization = 2 * ((fitness * precision) / (fitness + precision))

    if strategy == "mh":
        print("**** ", str(system), " Job ", str(job), " on PN ", str(pn_file), " using MH SAMPLING on suffix ", str(suffix)," ***")
    elif strategy == "naive":
        print("**** ", str(system), " Job ", str(job), " on PN ", str(pn_file), " using NAIVE SAMPLING on suffix ", str(suffix), " ***")
    else:
        raise ValueError("Unknown strategy.")
    print("AVATAR Generalization=", generalization)