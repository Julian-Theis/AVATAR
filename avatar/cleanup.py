import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import argparse
from conf.settings import DATA_PATH

WORK_PATH = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sfx', '--suffix', help='Suffix (selected epoch, e.g. 1981, to keep)', required=True)
    parser.add_argument('-s', '--system', help='System name', required=True)
    parser.add_argument('-j', '--job', help='Job (0,1)', required=True)

    args = parser.parse_args()

    system = args.system
    suffix = int(args.suffix)
    job = int(args.job)

    if DATA_PATH is None:
        directory = os.path.join(WORK_PATH, "data", "avatar", "sgans", system, str(job), "tf_logs", "ckpt")
    else:
        directory = os.path.join(DATA_PATH, "avatar", "sgans", system, str(job), "tf_logs", "ckpt")

    allfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    to_delete = []
    for f in allfiles:
        if f == "checkpoint" or "iw_dict_" in f:
            continue

        if ".pre_model-" in f:
            to_delete.append(f)
            continue

        if system in f and ".adv_model-" in f:
            index = int(str(f.split(system + ".adv_model-")[1]).split(".")[0])
            if index != suffix:
                to_delete.append(f)

    for f in tqdm(to_delete):
        os.remove(os.path.join(directory, f))

    print("Done.")