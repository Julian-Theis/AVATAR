from util.playout import standard_playout
import os
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__": 
    pn = "data/systems/pb_system_4_1.pnml"
    f_pop = "data/variants/pb_system_4_1_10_pop.txt"
    f_train = "data/variants/pb_system_4_1_10_train.txt"
    f_test = "data/variants/pb_system_4_1_10_test.txt"
    xes_train = "data/variants/pb_system_4_1_10_train.xes"
    csv_train = "data/variants/pb_system_4_1_10_train.csv"

    standard_playout(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train, train_size=0.1)