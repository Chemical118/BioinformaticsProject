import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from Bio import motifs
from Bio.Seq import Seq

import network
MakeSampleMotif = False

# basic input
# num_sample = int(input("Sample Number (Notice that a sample of twice the value you put in will be made) : "))
# seq_length = int(input("Sample Length : "))
# num_motif = int(input("Motif Number : "))
# mot_length = int(input("Motif Length : "))
# target_seq = Seq(input("Target Sequence : (Make sure shorter than your motif Length) : "))
num_sample = 100
seq_length = 20
num_motif = 200
mot_length = 6
target_seq = Seq("AGCAGC")
st = time.time()
tar_length = len(target_seq)
test_num_sample = 2 * num_sample

# check right input
if seq_length - tar_length - mot_length < 0 or tar_length != mot_length or num_sample < 0 or seq_length < 0 or num_motif < 0 or mot_length < 0:
    print("Wrong input Try again")
    sys.exit(1)

# make target motif
m = motifs.create([target_seq])
pwm = m.counts.normalize(pseudocounts=0)
pwm_arr = np.array(list(pwm.values()))
pwm = np.hstack((np.ones((4, mot_length)), pwm_arr, np.ones((4, seq_length - tar_length - mot_length))))

# make sample & motif
pos = np.array([np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=pwm[:, i] / sum(pwm[:, i])) for i in
                range(seq_length)]).transpose()
neg = np.array([np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=np.array([1, 1, 1, 1]) / 4) for i in
                range(seq_length)]).transpose()
test_pos = np.array([np.random.choice(['A', 'C', 'G', 'T'], test_num_sample, p=pwm[:, i] / sum(pwm[:, i])) for i in
                     range(seq_length)]).transpose()
test_neg = np.array([np.random.choice(['A', 'C', 'G', 'T'], test_num_sample, p=np.array([1, 1, 1, 1]) / 4) for i in
                     range(seq_length)]).transpose()
mot = np.array([np.random.choice(['A', 'C', 'G', 'T'], num_motif, p=np.array([1, 1, 1, 1]) / 4) for i in
                range(mot_length)]).transpose()

# if you want, you can make a motif that include target sequence
# change the value "MakeSampleMotif"
if MakeSampleMotif:
    for i in range(tar_length):
        mot[0][i] = str(target_seq[i])

# test one-hot encoding
base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
test_onehot_encode_pos = np.zeros((test_num_sample, seq_length, 4))
test_onehot_encode_pos_label = np.zeros((test_num_sample, 2), dtype=int)
test_onehot_encode_pos_label[:, 0] = 1
test_onehot_encode_neg = np.zeros((test_num_sample, seq_length, 4))
test_onehot_encode_neg_label = np.zeros((test_num_sample, 2), dtype=int)
test_onehot_encode_neg_label[:, 1] = 1
for i in range(test_num_sample):
    for j in range(seq_length):
        test_onehot_encode_pos[i, j, base_dict[test_pos[i, j]]] = 1
        test_onehot_encode_neg[i, j, base_dict[test_neg[i, j]]] = 1
test_x = np.vstack((test_onehot_encode_pos, test_onehot_encode_neg))
test_y_train = np.vstack((test_onehot_encode_pos_label, test_onehot_encode_neg_label))
test_num_sample *= 2  # cause sample is double!

# sample one-hot encoding
onehot_encode_pos = np.zeros((num_sample, seq_length, 4))
onehot_encode_pos_label = np.zeros((num_sample, 2), dtype=int)
onehot_encode_pos_label[:, 0] = 1
onehot_encode_neg = np.zeros((num_sample, seq_length, 4))
onehot_encode_neg_label = np.zeros((num_sample, 2), dtype=int)
onehot_encode_neg_label[:, 1] = 1
for i in range(num_sample):
    for j in range(seq_length):
        onehot_encode_pos[i, j, base_dict[pos[i, j]]] = 1
        onehot_encode_neg[i, j, base_dict[neg[i, j]]] = 1
x = np.vstack((onehot_encode_pos, onehot_encode_neg))
y_train = np.vstack((onehot_encode_pos_label, onehot_encode_neg_label))
num_sample *= 2  # cause sample is double!

# motif one-hot encoding
onehot_encode_mot = np.zeros((num_motif, mot_length, 4))
for i in range(num_motif):
    for j in range(mot_length):
        onehot_encode_mot[i, j, base_dict[mot[i, j]]] = 1

# etc calculation & definition
z = onehot_encode_mot
x_train = np.zeros((num_sample, num_motif), dtype=float)
test_x_train = np.zeros((test_num_sample, num_motif), dtype=float)

# make convolution
for i in range(num_sample):
    for j in range(num_motif):
        p = np.zeros((seq_length - mot_length + 1), dtype=float)
        for k in range(seq_length - mot_length + 1):
            for l in range(mot_length):
                p[k] += float(np.convolve(x[i][k + l], np.flip(z[j][l]), 'valid'))
        # x_train[i, j] = np.convolve(p, np.flip(p / sum(p)), 'valid')  # Stochastic pooling
        x_train[i, j] = np.max(p)  # max pooling

for i in range(test_num_sample):
    for j in range(num_motif):
        test_p = np.zeros((seq_length - mot_length + 1), dtype=float)
        for k in range(seq_length - mot_length + 1):
            for l in range(mot_length):
                test_p[k] += float(np.convolve(test_x[i][k + l], np.flip(z[j][l]), 'valid'))
        # test_x_train[i, j] = np.convolve(test_p, np.flip(test_p / sum(test_p)), 'valid')  # Stochastic pooling
        test_x_train[i, j] = np.max(test_p)  # max pooling

train_data = [[] for _ in range(num_sample)]
test_train_data = [[] for _ in range(test_num_sample)]
test_data = []

for i in range(num_sample):
    train_data[i].append(np.zeros((num_motif, 1)))
    train_data[i].append(np.zeros((2, 1)))
    for j in range(num_motif):
        train_data[i][0][j] = x_train[i, j]
    for j in range(2):
        train_data[i][1][j] = y_train[i, j]

for i in range(test_num_sample):
    test_train_data[i].append(np.zeros((num_motif, 1)))
    test_train_data[i].append(np.zeros((2, 1)))
    for j in range(num_motif):
        test_train_data[i][0][j] = test_x_train[i, j]
    for j in range(2):
        test_train_data[i][1][j] = test_y_train[i, j]
    test_data.append((test_train_data[i][0], int(test_train_data[i][1][1])))

print(train_data)
print(test_data)
net = network.Network([num_motif, int(1.5 * num_motif), 50, 10, 2])
net.SGD(train_data, 200, 60, 1.0, test_data=test_data)
print("execution time :", str(round(time.time() - st, 4)), "s")
plt.figure()
plt.plot(net.history())
plt.ylabel('val accuracy')
plt.xlabel('epoch')
plt.legend(['valid accuracy'])
plt.show()
