from tensorflow.keras import models
from compro import process
from datas import blosum62 as blo62
from datas import data_list
from datetime import datetime
import numpy as np
import random as r

pros = process()
data = data_list()
pro = pros[0]


def random_gene(sset, num=3):
    global sl_pro
    for st, s_se in enumerate(sset):
        s_se = s_se[0]
        for _ in range(num):
            sd = r.randint(0, len(s_se) - 1)
            s_se[sd] = r.choice(sl_pro[sd][0])
    return sset


tar = 4
size = 500

dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
dtot_list = list(filter(lambda t: t[1] > 9, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
loc_list = list(map(lambda t: t[0], dtot_list))
sl_pro = []
tot_iter = []
tot_iterl = []
base_list = []
for i in dtot_list:
    sl_pro.append(pro[i[0]] + [i[0]])  # ([[["M", "K"], {각각에 대한 딕셔너리}, 단백질 위치].. ])
for i in sl_pro:
    len_list = list(map(lambda t: len(t), i[1].values()))
    base_list.append(i[0][np.argmax(len_list)])
data.sort(key=lambda t: -t[1][tar])
seq_set = [[[], 0] for _ in range(3 * size)]
t_set = []
for i in range(3):
    for _ in range(size):
        t_set.append(data[i][0])
for ind, val in enumerate(t_set):
    for j in sl_pro:
        seq_set[ind][0].append(val[j[2]])
model = models.load_model("keras_rubisco")
nptest = np.array(range(10))
seq_set = random_gene(seq_set, num=2)
ans = seq_set[0]
cnt = 0
with open("ans.txt", "w") as f:
    now = datetime.now()
    f.write("%d-%02d-%02d %02d:%02d:%02d\n" %(now.year, now.month, now.day, now.hour, now.minute, now.second))
while True:
    cnt += 1
    train_data = np.zeros((len(seq_set), len(dtot_list)))
    for step, s_set in enumerate(seq_set):
        s_set = s_set[0]
        for ind, val in enumerate(s_set):
            train_data[step][ind] = blo62((base_list[ind], val))
    pre = model.predict(train_data)
    for idx, j in enumerate(pre):
        j = j / np.sum(j)
        test_val = j * nptest
        seq_set[idx][1] = sum(test_val)
    seq_set.sort(key=lambda t: -t[1])
    if ans[1] < seq_set[0][1]:
        ans = seq_set[0]
        ans_str = ""
        for ind, val in enumerate(pro):
            if ind in loc_list:
                ans_str += ans[0][loc_list.index(ind)]
            else:
                len_list = list(map(lambda t: len(t), val[1].values()))
                ans_str += val[0][np.argmax(len_list)]
        with open("ans.txt", "a") as f:
            f.write("%d %.5f %s\n" % (cnt, ans[1], ans_str))
    seq_set = seq_set[:20] + random_gene(seq_set[20:], num=3)
