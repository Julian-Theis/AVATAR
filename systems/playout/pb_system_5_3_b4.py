from util.playout import biased_playout_b4
import os
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    if DATA_PATH is None:
        pn = os.path.join(WORK_PATH, "data", "systems", "pb_system_5_3.pnml")
        f_pop = os.path.join(WORK_PATH, "data", "variants", "pb_system_5_3_b4_pop.txt")
        f_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_5_3_b4_train.txt")
        f_test = os.path.join(WORK_PATH, "data", "variants", "pb_system_5_3_b4_test.txt")
        xes_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_5_3_b4_train.xes")
        csv_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_5_3_b4_train.csv")
    else:
        pn = os.path.join(DATA_PATH, "systems", "pb_system_5_3.pnml")
        f_pop = os.path.join(DATA_PATH, "variants", "pb_system_5_3_b4_pop.txt")
        f_train = os.path.join(DATA_PATH, "variants", "pb_system_5_3_b4_train.txt")
        f_test = os.path.join(DATA_PATH, "variants", "pb_system_5_3_b4_test.txt")
        xes_train = os.path.join(DATA_PATH, "variants", "pb_system_5_3_b4_train.xes")
        csv_train = os.path.join(DATA_PATH, "variants", "pb_system_5_3_b4_train.csv")

    biased_playout_b4(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train)