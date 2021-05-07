from util.playout import biased_playout_b4

if __name__ == "__main__": 
    pn = "data/systems/pb_system_4_1.pnml"
    f_pop = "data/variants/pb_system_4_1_b4_pop.txt"
    f_train = "data/variants/pb_system_4_1_b4_train.txt"
    f_test = "data/variants/pb_system_4_1_b4_test.txt"
    xes_train = "data/variants/pb_system_4_1_b4_train.xes"
    csv_train = "data/variants/pb_system_4_1_b4_train.csv"

    biased_playout_b4(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train)