from util.playout import standard_playout
import os
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    if DATA_PATH is None:
        pn = os.path.join(WORK_PATH, "data", "systems", "pb_system_2_4.pnml")
        f_pop = os.path.join(WORK_PATH, "data", "variants", "pb_system_2_4_s50_pop.txt")
        f_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_2_4_s50_train.txt")
        f_test = os.path.join(WORK_PATH, "data", "variants", "pb_system_2_4_s50_test.txt")
        xes_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_2_4_s50_train.xes")
        csv_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_2_4_s50_train.csv")
    else:
        pn = os.path.join(DATA_PATH, "systems", "pb_system_2_4.pnml")
        f_pop = os.path.join(DATA_PATH, "variants", "pb_system_2_4_s50_pop.txt")
        f_train = os.path.join(DATA_PATH, "variants", "pb_system_2_4_s50_train.txt")
        f_test = os.path.join(DATA_PATH, "variants", "pb_system_2_4_s50_test.txt")
        xes_train = os.path.join(DATA_PATH, "variants", "pb_system_2_4_s50_train.xes")
        csv_train = os.path.join(DATA_PATH, "variants", "pb_system_2_4_s50_train.csv")

    standard_playout(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train, train_size=0.5)