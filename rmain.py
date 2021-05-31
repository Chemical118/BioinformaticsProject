import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from compro import process
from datas import blosum62 as blo62
from datas import data_list
from sklearn.metrics import r2_score as r2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import shuffle
pros = process()
data = data_list()
pro = pros[0]


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


# pros 구조
# (pro, d_list, dnum_list)
# pro : 순서별로 몇개의 아미노산 종류가 있는지, 그것이 어느 것에는 어떤 것이 있는지 확인 ([[["M", "K"], {각각에 대한 딕셔너리}].. ])
# d_list : 하나의 위치에 2개 이상의 아미노산을 가지는 대상들의 번호
# dnum_list : d_list에서 가장 적은 개수를 가지는 아미노산의 개수 (몇개의 대상에서 mutaion이 일어났는지 확인)

# data_list는 아래와 같이 반환한다.
# [[Seq 단백질 서열, [종 이름, kcat, Kc, Sc/o, Eff., type (A or B), EMBL code], index].. ]

dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
dtot_list = list(filter(lambda t: t[1] > 9, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
test_loca_list = list(map(lambda t: [t[0]], dtot_list))  # [[아미노산의 위치, motif 서열].. ]
# 원하는 값에 대해서 최대 최소 찾기
tar = 4  # 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
data.sort(key=lambda t: -t[1][tar])
data = data[1:]
shuffle(data)
tar_min = min(map(lambda t: t[1][tar], data))
tar_max = max(map(lambda t: t[1][tar], data))
num_data = len(data)
num_motif = len(test_loca_list)
train_data = np.zeros((num_data, num_motif))
train_label = np.zeros(num_data)

for ind, val in enumerate(test_loca_list):
    len_list = list(map(lambda t: len(t), pro[val[0]][1].values()))
    test_loca_list[ind].append(pro[val[0]][0][np.argmax(len_list)])

print(len(data))
for i, sdata in enumerate(data):
    for ind, val in enumerate(test_loca_list):
        pro_loc = val[0]
        pro_mot = val[1]
        train_data[i][ind] = blo62((pro_mot, sdata[0][pro_loc])) + 4  # 0 이상으로 변환
    tar_val = sdata[1][tar]
    tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
    train_label[i] = tar_val * 10

model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=[num_motif]),
    layers.Dense(30, activation='relu'),
    layers.Dense(15, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=['mae', 'mse'])
model.summary()
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(
    train_data, train_label,
    epochs=1000, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])
print()

loss, mae, mse = model.evaluate(train_data, train_label, verbose=2)
test_predictions = model.predict(train_data).flatten()

print("평균 절대 오차: {:5.2f}".format(mae))
print("모델의 R^2 : %.4f" % r2(train_label, test_predictions))
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
hist['epoch'] = history.epoch
plt.figure(figsize=(8, 12))
plt.subplot(2, 1, 1)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [Eff.]')
plt.plot(hist['epoch'], hist['mae'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'],
         label='Val Error')
plt.ylim([0, 5])
plt.legend()
plt.subplot(2, 1, 2)
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$Eff.^2$]')
plt.plot(hist['epoch'], hist['mse'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'],
         label='Val Error')
plt.ylim([0, 20])
plt.legend()
plt.show()

plt.scatter(train_label, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# with open("ans/ans.csv", "w", encoding="utf-8") as f:
#     for i in range(num_motif):
#         f.write(str(train_label[i]) + ',' + str(test_predictions[i]) + '\n')
# csv로 (실제 값, 예측 값) 인쇄
