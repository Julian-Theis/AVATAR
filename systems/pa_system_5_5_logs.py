from pm4py.objects.petri.importer import factory as pnml_importer
import os
from processmining.algorithm.playout import Player
from sklearn.model_selection import train_test_split
import numpy as np

def writeToFile(file, lst):
    with open(file, 'w') as outfile:
        for entry in lst:
            print_trace = ""
            for index, ev in enumerate(entry):
                if index == 0:
                    print_trace = str(ev).replace(" ", "")
                else:
                    print_trace = print_trace + " " + str(ev).replace(" ", "")
            outfile.write(print_trace.strip() + "\n")

if __name__ == "__main__": 
    pn = "pa_system_5_5.pnml"
    f_pop = "pa_system_5_5_pop.txt"
    f_train = "pa_system_5_5_train.txt"
    f_test = "pa_system_5_5_test.txt"

    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print(initial_marking)
    print(final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()


    print("*** ALL POSSIBLE TRACES ***")
    #for trace in gen_traces: print(trace)
    print(len(gen_traces))
    writeToFile(f_pop, gen_traces)

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

    writeToFile(f_train, train)
    writeToFile(f_test, test)