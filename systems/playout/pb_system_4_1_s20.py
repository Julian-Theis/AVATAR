from util.playout import standard_playout
import os
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    if DATA_PATH is None:
        pn = os.path.join(WORK_PATH, "data", "systems", "pb_system_4_1.pnml")
        f_pop = os.path.join(WORK_PATH, "data", "variants", "pb_system_4_1_s20_pop.txt")
        f_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_4_1_s20_train.txt")
        f_test = os.path.join(WORK_PATH, "data", "variants", "pb_system_4_1_s20_test.txt")
        xes_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_4_1_s20_train.xes")
        csv_train = os.path.join(WORK_PATH, "data", "variants", "pb_system_4_1_s20_train.csv")
    else:
        pn = os.path.join(DATA_PATH, "systems", "pb_system_4_1.pnml")
        f_pop = os.path.join(DATA_PATH, "variants", "pb_system_4_1_s20_pop.txt")
        f_train = os.path.join(DATA_PATH, "variants", "pb_system_4_1_s20_train.txt")
        f_test = os.path.join(DATA_PATH, "variants", "pb_system_4_1_s20_test.txt")
        xes_train = os.path.join(DATA_PATH, "variants", "pb_system_4_1_s20_train.xes")
        csv_train = os.path.join(DATA_PATH, "variants", "pb_system_4_1_s20_train.csv")

    standard_playout(pn=pn, f_pop=f_pop, f_train=f_train, f_test=f_test, xes_train=xes_train, csv_train=csv_train, train_size=0.2)