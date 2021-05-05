from random import choice

import matplotlib.pyplot as plt
import numpy as np

import network
from compro import process
from datas import blosum62 as blo62
from datas import data_list
from datas import nums

pros = process()
data = data_list()
pro = pros[0]

# pros 구조
# (pro, d_list, dnum_list)
# pro : 순서별로 몇개의 아미노산 종류가 있는지, 그것이 어느 것에는 어떤 것이 있는지 확인 ([[["M", "K"], {각각에 대한 딕셔너리}].. ])
# d_list : 하나의 위치에 2개 이상의 아미노산을 가지는 대상들의 번호
# dnum_list : d_list에서 가장 적은 개수를 가지는 아미노산의 개수 (몇개의 대상에서 mutaion이 일어났는지 확인)

# data_list는 아래와 같이 반환한다.
# [[Seq 단백질 서열, [종 이름, kcat, Kc, Sc/o, Eff., type (A or B), EMBL code], index].. ]

dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
dtot_list = list(filter(lambda t: t[1] > 8, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
test_loca_list = list(map(lambda t: [t[0]], dtot_list))  # [[아미노산의 위치, motif 서열].. ]
train_data = [[] for _ in range(len(data))]
test_data = []

for ind, val in enumerate(test_loca_list):
    test_loca_list[ind].append(choice(pro[ind][0]))

# 원하는 값에 대해서 최대 최소 찾기
tar = 4  # 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
tar_min = min(map(lambda t: t[1][tar], data))
tar_max = max(map(lambda t: t[1][tar], data))

for i, sdata in enumerate(data):
    train_data[i].append(np.zeros((len(test_loca_list), 1)))
    train_data[i].append(np.zeros((10, 1)))
    for ind, val in enumerate(test_loca_list):
        pro_loc = val[0]
        pro_mot = val[1]
        train_data[i][0][ind] = blo62((pro_mot, sdata[0][pro_loc]))
        tar_val = sdata[1][tar]
        tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
        train_data[i][1][nums(tar_ind)] = 1

for i in train_data:
    test_data.append((i[0], np.argmax(i[1])))

num_motif = len(test_loca_list)
print(num_motif)

net = network.Network([num_motif, int(1.5 * num_motif), 25, 10])
net.SGD(train_data, 2000, 20, 0.7, test_data=test_data)
plt.figure()
plt.plot(net.history())
plt.ylabel('val accuracy')
plt.xlabel('epoch')
plt.legend(['valid accuracy'])
plt.show()
