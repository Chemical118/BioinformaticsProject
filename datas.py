def data_list():
    """
    df_list는 아래와 같은 모습을 가진다.
    [[Seq 단백질 서열, [종 이름, kcat, Kc, Sc/o, Eff., type (A or B), EMBL code], index]
    """
    from Bio import SeqIO
    import pandas as pd
    df = pd.read_excel('data.xls')
    df_list = df.values.tolist()
    df_list = list(map(lambda t: [*t[::-1]], enumerate(df_list)))
    for i in range(75):
        df_list[i].insert(0, SeqIO.read("data/" + str(i) + ".fasta", "fasta").seq)
    return df_list


def blosum62(pair):
    """
    순서에 상관없이 matrix를 이용하도록 간단한 in을 이용해서 사용하는 함수
    pair은 2개의 tuple로 넣어주어야 한다
    """
    matrix = {
        ("W", "F"): 1, ("L", "R"): -2, ("S", "P"): -1, ("V", "T"): 0,
        ("Q", "Q"): 5, ("N", "A"): -2, ("Z", "Y"): -2, ("W", "R"): -3,
        ("Q", "A"): -1, ("S", "D"): 0, ("H", "H"): 8, ("S", "H"): -1,
        ("H", "D"): -1, ("L", "N"): -3, ("W", "A"): -3, ("Y", "M"): -1,
        ("G", "R"): -2, ("Y", "I"): -1, ("Y", "E"): -2, ("B", "Y"): -3,
        ("Y", "A"): -2, ("V", "D"): -3, ("B", "S"): 0, ("Y", "Y"): 7,
        ("G", "N"): 0, ("E", "C"): -4, ("Y", "Q"): -1, ("Z", "Z"): 4,
        ("V", "A"): 0, ("C", "C"): 9, ("M", "R"): -1, ("V", "E"): -2,
        ("T", "N"): 0, ("P", "P"): 7, ("V", "I"): 3, ("V", "S"): -2,
        ("Z", "P"): -1, ("V", "M"): 1, ("T", "F"): -2, ("V", "Q"): -2,
        ("K", "K"): 5, ("P", "D"): -1, ("I", "H"): -3, ("I", "D"): -3,
        ("T", "R"): -1, ("P", "L"): -3, ("K", "G"): -2, ("M", "N"): -2,
        ("P", "H"): -2, ("F", "Q"): -3, ("Z", "G"): -2, ("X", "L"): -1,
        ("T", "M"): -1, ("Z", "C"): -3, ("X", "H"): -1, ("D", "R"): -2,
        ("B", "W"): -4, ("X", "D"): -1, ("Z", "K"): 1, ("F", "A"): -2,
        ("Z", "W"): -3, ("F", "E"): -3, ("D", "N"): 1, ("B", "K"): 0,
        ("X", "X"): -1, ("F", "I"): 0, ("B", "G"): -1, ("X", "T"): 0,
        ("F", "M"): 0, ("B", "C"): -3, ("Z", "I"): -3, ("Z", "V"): -2,
        ("S", "S"): 4, ("L", "Q"): -2, ("W", "E"): -3, ("Q", "R"): 1,
        ("N", "N"): 6, ("W", "M"): -1, ("Q", "C"): -3, ("W", "I"): -3,
        ("S", "C"): -1, ("L", "A"): -1, ("S", "G"): 0, ("L", "E"): -3,
        ("W", "Q"): -2, ("H", "G"): -2, ("S", "K"): 0, ("Q", "N"): 0,
        ("N", "R"): 0, ("H", "C"): -3, ("Y", "N"): -2, ("G", "Q"): -2,
        ("Y", "F"): 3, ("C", "A"): 0, ("V", "L"): 1, ("G", "E"): -2,
        ("G", "A"): 0, ("K", "R"): 2, ("E", "D"): 2, ("Y", "R"): -2,
        ("M", "Q"): 0, ("T", "I"): -1, ("C", "D"): -3, ("V", "F"): -1,
        ("T", "A"): 0, ("T", "P"): -1, ("B", "P"): -2, ("T", "E"): -1,
        ("V", "N"): -3, ("P", "G"): -2, ("M", "A"): -1, ("K", "H"): -1,
        ("V", "R"): -3, ("P", "C"): -3, ("M", "E"): -2, ("K", "L"): -2,
        ("V", "V"): 4, ("M", "I"): 1, ("T", "Q"): -1, ("I", "G"): -4,
        ("P", "K"): -1, ("M", "M"): 5, ("K", "D"): -1, ("I", "C"): -1,
        ("Z", "D"): 1, ("F", "R"): -3, ("X", "K"): -1, ("Q", "D"): 0,
        ("X", "G"): -1, ("Z", "L"): -3, ("X", "C"): -2, ("Z", "H"): 0,
        ("B", "L"): -4, ("B", "H"): 0, ("F", "F"): 6, ("X", "W"): -2,
        ("B", "D"): 4, ("D", "A"): -2, ("S", "L"): -2, ("X", "S"): 0,
        ("F", "N"): -3, ("S", "R"): -1, ("W", "D"): -4, ("V", "Y"): -1,
        ("W", "L"): -2, ("H", "R"): 0, ("W", "H"): -2, ("H", "N"): 1,
        ("W", "T"): -2, ("T", "T"): 5, ("S", "F"): -2, ("W", "P"): -4,
        ("L", "D"): -4, ("B", "I"): -3, ("L", "H"): -3, ("S", "N"): 1,
        ("B", "T"): -1, ("L", "L"): 4, ("Y", "K"): -2, ("E", "Q"): 2,
        ("Y", "G"): -3, ("Z", "S"): 0, ("Y", "C"): -2, ("G", "D"): -1,
        ("B", "V"): -3, ("E", "A"): -1, ("Y", "W"): 2, ("E", "E"): 5,
        ("Y", "S"): -2, ("C", "N"): -3, ("V", "C"): -1, ("T", "H"): -2,
        ("P", "R"): -2, ("V", "G"): -3, ("T", "L"): -1, ("V", "K"): -2,
        ("K", "Q"): 1, ("R", "A"): -1, ("I", "R"): -3, ("T", "D"): -1,
        ("P", "F"): -4, ("I", "N"): -3, ("K", "I"): -3, ("M", "D"): -3,
        ("V", "W"): -3, ("W", "W"): 11, ("M", "H"): -2, ("P", "N"): -2,
        ("K", "A"): -1, ("M", "L"): 2, ("K", "E"): 1, ("Z", "E"): 4,
        ("X", "N"): -1, ("Z", "A"): -1, ("Z", "M"): -1, ("X", "F"): -1,
        ("K", "C"): -3, ("B", "Q"): 0, ("X", "B"): -1, ("B", "M"): -3,
        ("F", "C"): -2, ("Z", "Q"): 3, ("X", "Z"): -1, ("F", "G"): -3,
        ("B", "E"): 1, ("X", "V"): -1, ("F", "K"): -3, ("B", "A"): -2,
        ("X", "R"): -1, ("D", "D"): 6, ("W", "G"): -2, ("Z", "F"): -3,
        ("S", "Q"): 0, ("W", "C"): -2, ("W", "K"): -3, ("H", "Q"): 0,
        ("L", "C"): -1, ("W", "N"): -4, ("S", "A"): 1, ("L", "G"): -4,
        ("W", "S"): -3, ("S", "E"): 0, ("H", "E"): 0, ("S", "I"): -2,
        ("H", "A"): -2, ("S", "M"): -1, ("Y", "L"): -1, ("Y", "H"): 2,
        ("Y", "D"): -3, ("E", "R"): 0, ("X", "P"): -2, ("G", "G"): 6,
        ("G", "C"): -3, ("E", "N"): 0, ("Y", "T"): -2, ("Y", "P"): -3,
        ("T", "K"): -1, ("A", "A"): 4, ("P", "Q"): -1, ("T", "C"): -1,
        ("V", "H"): -3, ("T", "G"): -2, ("I", "Q"): -3, ("Z", "T"): -1,
        ("C", "R"): -3, ("V", "P"): -2, ("P", "E"): -1, ("M", "C"): -1,
        ("K", "N"): 0, ("I", "I"): 4, ("P", "A"): -1, ("M", "G"): -3,
        ("T", "S"): 1, ("I", "E"): -3, ("P", "M"): -2, ("M", "K"): -1,
        ("I", "A"): -1, ("P", "I"): -3, ("R", "R"): 5, ("X", "M"): -1,
        ("L", "I"): 2, ("X", "I"): -1, ("Z", "B"): 1, ("X", "E"): -1,
        ("Z", "N"): 0, ("X", "A"): 0, ("B", "R"): -1, ("B", "N"): 3,
        ("F", "D"): -3, ("X", "Y"): -1, ("Z", "R"): 0, ("F", "H"): -1,
        ("B", "F"): -3, ("F", "L"): 0, ("X", "Q"): -1, ("B", "B"): 4
    }
    if pair in matrix:
        return matrix[pair]
    else:
        return matrix[pair[::-1]]


def nums(a):
    """
    score을 0부터 1까지로 나누었을 10개의 등급으로 나누어 주는 함수
    입력은 0부터 1까지의 실수
    출력은 0부터 9까지의 정수이다
    """
    if 0 <= a < 0.1:
        return 0
    elif 0.1 <= a < 0.2:
        return 1
    elif 0.2 <= a < 0.3:
        return 2
    elif 0.3 <= a < 0.4:
        return 3
    elif 0.4 <= a < 0.5:
        return 4
    elif 0.5 <= a < 0.6:
        return 5
    elif 0.6 <= a < 0.7:
        return 6
    elif 0.7 <= a < 0.8:
        return 7
    elif 0.8 <= a < 0.9:
        return 8
    else:
        return 9


def fkmain(epochs=2000, tar=4):
    """
    ["아미노산 서열 위치", "아미노산 서열", ["제곱의 합", "절댓값의 합", "최대값"]]
    """
    from tensorflow.keras import models
    from tensorflow.keras import layers
    import pickle
    import numpy as np

    with open("data/saved/data_list.pickle", "rb") as f:
        data = pickle.load(f)
    with open("data/saved/process.pickle", "rb") as f:
        pros = pickle.load(f)
    pro = pros[0]

    dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
    dtot_list = list(filter(lambda t: t[1] > 9, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
    test_loca_list = list(map(lambda t: [t[0]], dtot_list))  # [[아미노산의 위치, motif 서열].. ]
    num_data = len(data)
    num_motif = len(test_loca_list)
    train_data = np.zeros((num_data, num_motif))
    train_label = np.zeros((num_data, 10))

    for ind, val in enumerate(test_loca_list):
        len_list = list(map(lambda t: len(t), pro[val[0]][1].values()))
        test_loca_list[ind].append(pro[val[0]][0][np.argmax(len_list)])

    # 원하는 값에 대해서 최대 최소 찾기
    tar = tar  # 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
    tar_min = min(map(lambda t: t[1][tar], data))
    tar_max = max(map(lambda t: t[1][tar], data))

    for i, sdata in enumerate(data):
        for ind, val in enumerate(test_loca_list):
            pro_loc = val[0]
            pro_mot = val[1]
            train_data[i][ind] = blosum62((pro_mot, sdata[0][pro_loc]))
        tar_val = sdata[1][tar]
        tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
        train_label[i][nums(tar_ind)] = 1

    model = models.Sequential(name='BioInfoCNN')
    model.add(layers.Dense(30, activation='sigmoid', input_shape=(num_motif,)))
    model.add(layers.Dense(10, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_label, epochs=epochs, batch_size=None, verbose=0)
    model_weight = model.get_weights()
    for idx, mo in enumerate(model_weight[0]):
        test_loca_list[idx].append([])
        test_loca_list[idx][-1].append(sum(map(lambda t: t * t, mo)))
        test_loca_list[idx][-1].append(sum(map(lambda t: abs(t), mo)))
        test_loca_list[idx][-1].append(max(mo))

    return test_loca_list


def frmain(epochs=2000, tar=4):
    """
    ["아미노산 서열 위치", "아미노산 서열", ["제곱의 합", "절댓값의 합", "최대값"]]
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    from random import shuffle
    import pickle

    with open("data/saved/data_list.pickle", "rb") as f:
        data = pickle.load(f)
    with open("data/saved/process.pickle", "rb") as f:
        pros = pickle.load(f)
    pro = pros[0]

    dtot_list = list(zip(pros[1], pros[2]))  # dtot_list : [(아미노산 위치, mutation 개수).. ]
    dtot_list = list(filter(lambda t: t[1] > 9, dtot_list))  # 8개 이상의 mutaion을 가지는 dtot_list
    test_loca_list = list(map(lambda t: [t[0]], dtot_list))  # [[아미노산의 위치, motif 서열].. ]
    tar = tar  # 1 : kcat, 2 : Kc, 3 : Sc/o, 4 : Eff.
    shuffle(data)
    # tar_min = min(map(lambda t: t[1][tar], data))
    # tar_max = max(map(lambda t: t[1][tar], data))
    num_data = len(data)
    num_motif = len(test_loca_list)
    train_data = np.zeros((num_data, num_motif))
    train_label = np.zeros(num_data)

    for ind, val in enumerate(test_loca_list):
        len_list = list(map(lambda t: len(t), pro[val[0]][1].values()))
        test_loca_list[ind].append(pro[val[0]][0][np.argmax(len_list)])

    for i, sdata in enumerate(data):
        for ind, val in enumerate(test_loca_list):
            pro_loc = val[0]
            pro_mot = val[1]
            train_data[i][ind] = blosum62((pro_mot, sdata[0][pro_loc])) + 4  # 0 이상으로 변환
        tar_val = sdata[1][tar]
        # tar_ind = (tar_val - tar_min) / (tar_max - tar_min)
        train_label[i] = tar_val * 10

    model = keras.Sequential(name='BioInfoReg')
    model.add(layers.Dense(50, activation='relu', input_shape=(num_motif,)))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.RMSprop(0.0015),
                  metrics=['mae', 'mse'])
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    model.fit(train_data, train_label,
              epochs=epochs, verbose=0)

    model_weight = model.get_weights()
    for idx, mo in enumerate(model_weight[0]):
        test_loca_list[idx].append([])
        test_loca_list[idx][-1].append(sum(map(lambda t: t * t, mo)))
        test_loca_list[idx][-1].append(sum(map(lambda t: abs(t), mo)))
        test_loca_list[idx][-1].append(max(mo))

    return test_loca_list
