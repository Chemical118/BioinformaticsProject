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

loc_list = [142, 183, 189, 262, 449, 14, 95, 99, 251, 228]   # 해당 위치의 motif
loc_list = list(map(lambda t: t - 1, loc_list)) # 위치를 Python 기준으로 수정함
dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
dtot_list = list(filter(lambda t: t[0] in loc_list, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
test_loca_list = list(map(lambda t: [t[0]], dtot_list))  # [[아미노산의 위치, motif 서열].. ]
print(test_loca_list)
train_data = [[] for _ in range(len(data))]
test_data = []
for ind, val in enumerate(test_loca_list):
    len_list = list(map(lambda t: len(t), pro[ind][1].values()))
    test_loca_list[ind].append(pro[ind][0][np.argmax(len_list)])

num_motif = len(test_loca_list)
bas_weight = [num_motif, int(1.5 * num_motif), 10]


def deeplearn(tar, net_weight, epochs=2000, mini_batch_size=15, eta=0.5, silent=False):
    """
    tar => 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
    net_weight는 list로 반환한다, 맨 처음은 num_motif로 한다.
    그 아래 값은 대부분 기본적인 값을 따른다, 많은 실험을 통해서 결정된 값들이다.
    silent는 조용히 할지를 결정 할 수 있다. 기본은 False이다.
    """
    # 원하는 값에 대해서 최대 최소 찾기
    tar_min = min(map(lambda t: t[1][tar], data))
    tar_max = max(map(lambda t: t[1][tar], data))

    for i, sdata in enumerate(data):
        train_data[i].append(np.zeros((len(test_loca_list), 1)))
        train_data[i].append(np.zeros((10, 1)))
        for idx, va in enumerate(test_loca_list):
            pro_loc = va[0]
            pro_mot = va[1]
            train_data[i][0][idx] = blo62((pro_mot, sdata[0][pro_loc]))
            tar_val = sdata[1][tar]
            tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
            train_data[i][1][nums(tar_ind)] = 1

    for i in train_data:
        test_data.append((i[0], np.argmax(i[1])))

    net = network.Network(net_weight)
    net.SGD(train_data, epochs, mini_batch_size, eta, silent, test_data=test_data)
    plt.figure()
    plt.plot(list(map(lambda t: t / len(train_data), net.history())))
    plt.ylabel('val accuracy')
    plt.xlabel('epoch')
    plt.legend(['valid accuracy'])
    plt.show()


deeplearn(4, bas_weight, epochs=3000, silent=False)
