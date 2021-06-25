import pickle

pick_path = "data/saved/comp.pickle"
with open(pick_path, "rb") as f:
    r_list, k_list, num = pickle.load(f)
    print("-----\n%d\n-----" % num)
t = r_list