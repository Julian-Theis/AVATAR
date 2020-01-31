from util.playout import standard_playout

if __name__ == "__main__": 
    pn = "data/systems/pa_system_9_1.pnml"
    f_pop = "data/variants/pa_system_9_1_pop.txt"
    f_train = "data/variants/pa_system_9_1_train.txt"
    f_test = "data/variants/pa_system_9_1_test.txt"
    xes_train = "data/variants/pa_system_9_1_train.xes"
    csv_train = "data/variants/pa_system_9_1_train.csv"

    standard_playout(pn=pn, f_pop=f_pop, f_train=f_train,f_test=f_test, xes_train=xes_train, csv_train=csv_train)