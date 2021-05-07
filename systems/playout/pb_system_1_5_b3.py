from util.playout import biased_playout_b3

if __name__ == "__main__": 
    pn = "data/systems/pb_system_1_5.pnml"
    f_pop = "data/variants/pb_system_1_5_b3_pop.txt"
    f_train = "data/variants/pb_system_1_5_b3_train.txt"
    f_test = "data/variants/pb_system_1_5_b3_test.txt"
    xes_train = "data/variants/pb_system_1_5_b3_train.xes"
    csv_train = "data/variants/pb_system_1_5_b3_train.csv"

    biased_playout_b3(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train)