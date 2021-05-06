from random import choice

import matplotlib.pyplot as plt
import numpy as np

from keras import models
from keras import layers
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
num_data = len(data)
num_motif = len(test_loca_list)
train_data = np.zeros((num_data, num_motif))
train_label = np.zeros((num_data, 10))

for ind, val in enumerate(test_loca_list):
    test_loca_list[ind].append(choice(pro[ind][0]))

# 원하는 값에 대해서 최대 최소 찾기
tar = 4  # 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
tar_min = min(map(lambda t: t[1][tar], data))
tar_max = max(map(lambda t: t[1][tar], data))

for i, sdata in enumerate(data):
    for ind, val in enumerate(test_loca_list):
        pro_loc = val[0]
        pro_mot = val[1]
        train_data[i][ind] = blo62((pro_mot, sdata[0][pro_loc]))
    tar_val = sdata[1][tar]
    tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
    train_label[i][nums(tar_ind)] = 1

model = models.Sequential()
model.add(layers.Dense(30, activation='sigmoid', input_shape=(num_motif,)))
model.add(layers.Dense(15, activation='sigmoid'))
model.add(layers.Dense(10, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_data, train_label, epochs=10000, batch_size=None)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('valid accuracy and valid loss')
plt.ylabel('val accuracy and val loss')
plt.xlabel('epoch')
plt.legend(['valid loss', 'valid accuracy'])
plt.show()

test_loss, test_acc = model.evaluate(train_data, train_label)
print('test_acc: ', test_acc)
