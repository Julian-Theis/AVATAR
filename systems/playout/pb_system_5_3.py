from util.playout import standard_playout

if __name__ == "__main__": 
    pn = "data/systems/pb_system_5_3.pnml"
    f_pop = "data/variants/pb_system_5_3_pop.txt"
    f_train = "data/variants/pb_system_5_3_train.txt"
    f_test = "data/variants/pb_system_5_3_test.txt"
    xes_train = "data/variants/pb_system_5_3_train.xes"
    csv_train = "data/variants/pb_system_5_3_train.csv"

    standard_playout(pn=pn, f_pop=f_pop, f_train=f_train,f_test=f_test, xes_train=xes_train, csv_train=csv_train)