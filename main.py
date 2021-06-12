import pickle
with open("data/saved/comp.pickle", "rb") as f:
    r_list, k_list = pickle.load(f)
for i in range(3):
    r_list.sort(key=lambda t: -t[2][i])
    k_list.sort(key=lambda t: -t[2][i])
    for ind, val in enumerate(k_list):
        print(ind + 1, list(map(lambda t: t[0], r_list)).index(val[0]) + 1)
    print("--------")
