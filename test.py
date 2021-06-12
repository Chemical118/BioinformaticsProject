from datas import frmain
from datas import fkmain
import pickle
r_list = frmain()
k_list = fkmain()
for i in range(100):
    for ind, val in enumerate(frmain()):
        for j in range(3):
            r_list[ind][-1][j] += val[-1][j]
    for ind, val in enumerate(fkmain()):
        for j in range(3):
            k_list[ind][-1][j] += val[-1][j]
    print(i)
with open("data/saved/comp.pickle", "wb") as f:
    pickle.dump((r_list, k_list), f)
