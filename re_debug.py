from tensorflow.keras import models
from datas import blosum62 as blo62
import numpy as np

loc = [13, 27, 42, 85, 90, 93, 94, 96, 98, 141, 142, 144, 188, 218, 224, 225, 227, 250, 269, 278, 319, 325, 327, 340,
       353, 358, 362, 370, 374, 417, 446, 448, 460]
base_list = ['K', 'E', 'T', 'H', 'A', 'E', 'N', 'Y', 'A', 'P', 'A', 'S', 'C', 'V', 'I', 'Y', 'A', 'I', 'L', 'S', 'M',
             'I', 'A', 'I', 'I', 'S', 'F', 'M', 'I', 'A', 'E', 'C', 'V']

test = "QETGVENYCITACVIYSIISMISSTSFMIVEGI"


def regde(t):
    len_data = len(loc)
    train_data = np.zeros((1, len_data))
    if len(t) != len_data:
        try:
            t = "".join(t[i] for i in loc)
        except IndexError:
            print("wrong input")
            exit(-1)
    model = models.load_model("reg_rubisco")
    for ind, val in enumerate(t):
        train_data[0][ind] = blo62((val, base_list[ind])) + 4
    return model.__call__(train_data).numpy().item()


if __name__ == '__main__':
    print(regde(test))
