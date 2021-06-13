with open("ans/kmain.txt", "r", encoding='utf-8') as f:
    cn_list = list(map(lambda t: t[:-1], f.readlines()))
with open("ans/rmain.txt", "r", encoding='utf-8') as f:
    rg_list = list(map(lambda t: t[:-1], f.readlines()))
tar = "-------------"

for ind, val in enumerate(cn_list):
    if val == tar:
        cn_list = cn_list[ind + 1:]
        break
for ind, val in enumerate(rg_list):
    if val == tar:
        rg_list = rg_list[ind + 1:]
        break
cn_list = list(map(lambda t: int(t.split()[0]), cn_list))
rg_list = list(map(lambda t: int(t.split()[0]), rg_list))

with open("ans/compare.csv", "w", encoding="utf-8") as f:
    for ind, val in enumerate(cn_list):
        print(ind + 1, rg_list.index(val) + 1)
        f.write("%d,%d\n" % (ind + 1, rg_list.index(val) + 1))
