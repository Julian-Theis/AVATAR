import os
import argparse

from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.evaluation.generalization import factory as generalization_factory
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

def main(system, miner):
    if DATA_PATH is None:
        log = xes_importer.import_log(os.path.join(WORK_PATH, "data", "variants", str(system) + "_train.xes"))
    else:
        log = xes_importer.import_log(os.path.join(DATA_PATH, "variants", str(system) + "_train.xes"))

    bestmodel = None
    bestfit = None
    bestPrec = None
    bestGen = 0
    bestfittraces = 0

    gen_bestmodel = None
    gen_bestfit = None
    gen_bestPrec = None
    gen_bestGen = 0

    if DATA_PATH is None:
        dir = os.listdir(os.path.join(WORK_PATH, "data", "pns", str(system)))
    else:
        dir = os.listdir(os.path.join(DATA_PATH, "pns", str(system)))

    for file in dir:
        if system in file and miner in file:
            if DATA_PATH is None:
                path = os.path.join(WORK_PATH, "data", "pns", str(system), file)
            else:
                path = os.path.join(DATA_PATH, "pns", str(system), file)

            print("Checking conformance of file:", path)

            net, initial_marking, final_marking = pnml_importer.import_net(path)

            fitness = replay_factory.apply(log, net, initial_marking, final_marking)
            precision = precision_factory.apply(log, net, initial_marking, final_marking)
            generalization = generalization_factory.apply(log, net, initial_marking, final_marking)

            if fitness['perc_fit_traces'] > bestfittraces:
                bestfittraces = fitness['perc_fit_traces']
                bestmodel = path
                bestfit = fitness
                bestPrec = precision
                bestGen = generalization

            elif generalization > bestGen and fitness['perc_fit_traces'] == bestfittraces:
                bestmodel = path
                bestfit = fitness
                bestPrec = precision
                bestGen = generalization

            if generalization > gen_bestGen:
                gen_bestmodel = path
                gen_bestfit = fitness
                gen_bestPrec = precision
                gen_bestGen = generalization


    net, initial_marking, final_marking = pnml_importer.import_net(gen_bestmodel)
    try:
        align_fitness = replay_factory.apply(log, net, initial_marking, final_marking, variant="alignments")
    except:
        align_fitness = {"averageFitness" : "N/A"}
    try:
        align_precision = precision_factory.apply(log, net, initial_marking, final_marking, variant="align_etconformance")
    except:
        align_precision = "N/A"
    print("")
    print("")
    print("*********** Petri net w/ highest ratio of fitting traces and high generalization *************** ")
    print("Petri net file:", gen_bestmodel)
    print("Token-based Fitness=", gen_bestfit['average_trace_fitness'])
    print("Token-based Precision=", gen_bestPrec)
    print("Alignment-based Fitness=",align_fitness['averageFitness'])
    print("Alignment-based Precision=", align_precision)
    print("Generalization=", gen_bestGen)

    net, initial_marking, final_marking = pnml_importer.import_net(bestmodel)
    try:
        align_fitness = replay_factory.apply(log, net, initial_marking, final_marking, variant="alignments")
    except:
        align_fitness = {"averageFitness" : "N/A"}

    try:
        align_precision = precision_factory.apply(log, net, initial_marking, final_marking,variant="align_etconformance")
    except:
        align_precision = "N/A"
    print("")
    print("*********** Petri net w/ highest ratio of fitting traces and high generalization *************** ")
    print("Petri net file:", bestmodel)
    print("Token-based Fitness=", bestfit['average_trace_fitness'])
    print("Token-based Precision=", bestPrec)
    print("Alignment-based Fitness=",align_fitness['averageFitness'])
    print("Alignment-based Precision=", align_precision)
    print("Generalization=", bestGen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', help='Which system? For example: pb_system_5_3', required=True)
    parser.add_argument('-m', '--miner', help='Which miner (splitminer, fodina)?', required=True)
    args = parser.parse_args()
    main(args.system, args.miner)