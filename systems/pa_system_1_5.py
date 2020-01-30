import os

from pm4py.objects.petri.importer import factory as pnml_importer
from sklearn.model_selection import train_test_split

from processmining.playout import Player
from systems.util import writeToFile

if __name__ == "__main__": 
    pn = "system/pa_system_1_5.pnml"
    f_pop = "variants/pa_system_1_5_pop.txt"
    f_train = "variants/pa_system_1_5_train.txt"
    f_test = "variants/pa_system_1_5_test.txt"

    f_pn = os.path.join(pn)
    net, initial_marking, final_marking = pnml_importer.apply(f_pn)
    print(initial_marking)
    print(final_marking)

    player = Player(net, initial_marking, final_marking, maxTraceLength=200, rep_inv_thresh=100, max_loop=3)
    gen_traces = player.play()


    print("*** ALL POSSIBLE TRACES ***")
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