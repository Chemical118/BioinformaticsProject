import pickle
with open("data/saved/comp.pickle", "rb") as f:
    r_list, k_list = pickle.load(f)

ref = [9, 14, 31, 86, 95, 97, 99, 142, 145, 149, 183, 189, 251, 255, 256, 262, 281, 328, 439, 449]  # 논문에 있는 자리

for i in range(3):
    r_list.sort(key=lambda t: -t[2][i])
    k_list.sort(key=lambda t: -t[2][i])
    for ind, val in enumerate(k_list):
        st = ""
        if val[0] + 1 in ref:
            st = "★"
        print(ind + 1, list(map(lambda t: t[0], r_list)).index(val[0]) + 1, st)
    print("--------")
