from re_debug import regde
from ke_debug import kerde
loc = [13, 27, 42, 85, 90, 93, 94, 96, 98, 141, 142, 144, 188, 218, 224, 225, 227, 250, 269, 278, 319, 325, 327, 340,
       353, 358, 362, 370, 374, 417, 446, 448, 460]

with open("ans/ans.txt", "r", encoding='utf-8') as f:
    text_list = f.readlines()
with open("ans/rans.txt", "r", encoding='utf-8') as f:
    rtext_list = f.readlines()
for ind, val in enumerate(text_list):
    if ind:
        t = val.split()
        tar = "".join(t[-1][j] for j in loc)
        print(" ".join(t[:-1] + [tar]), end=" ")
        print(regde(tar))
    else:
        print(val, end="")

for ind, val in enumerate(rtext_list):
    if ind:
        t = val.split()
        tar = "".join(t[-1][j] for j in loc)
        print(" ".join(t[:-1] + [tar]), end=" ")
        print(kerde(tar))
    else:
        print(val, end="")
