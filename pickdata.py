import pickle
from datas import data_list
from compro import process

with open("data/saved/data_list.pickle", "wb") as f:
    pickle.dump(data_list(), f)
with open("data/saved/process.pickle", "wb") as f:
    pickle.dump(process(), f)
