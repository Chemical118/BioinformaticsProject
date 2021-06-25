from datas import frmain
from datas import fkmain
import os.path
import pickle
pick_path = "data/saved/comp.pickle"
num = 0

tar = 10
repeat = 2
if os.path.isfile(pick_path):
    with open(pick_path, "rb") as f:
        r_list, k_list, num = pickle.load(f)
        print("-----\n%d\n-----" % num)
else:
    r_list = frmain()
    k_list = fkmain()
    num += 1
for z in range(repeat):
    for i in range(tar):
        for ind, val in enumerate(frmain()):
            for j in range(3):
                r_list[ind][-1][j] += val[-1][j]
        for ind, val in enumerate(fkmain()):
            for j in range(3):
                k_list[ind][-1][j] += val[-1][j]
        print(i)
    num += tar
    with open(pick_path, "wb") as f:
        pickle.dump((r_list, k_list, num), f)
    print("------------\n%d\n------------" % z)
