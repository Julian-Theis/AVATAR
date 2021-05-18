from util.playout import biased_playout_b1
import os
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__": 
    pn = "data/systems/pb_system_4_1.pnml"
    f_pop = "data/variants/pb_system_4_1_b1_pop.txt"
    f_train = "data/variants/pb_system_4_1_b1_train.txt"
    f_test = "data/variants/pb_system_4_1_b1_test.txt"
    xes_train = "data/variants/pb_system_4_1_b1_train.xes"
    csv_train = "data/variants/pb_system_4_1_b1_train.csv"

    biased_playout_b1(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train)