import matplotlib.pyplot as plt
from matplotlib import rc

def process():
    from datas import data_list
    data = data_list()
    data = list(map(lambda t: t[0], data)) # 단순 단백질 서열 list로 변경
    len_data = len(data[0])  # data의 모든 길이는 같다
    pro = list()
    for i in range(len_data):
        data_set = list()
        data_dict = dict()
        for ind, val in enumerate(data):
            tar = str(val[i])
            if tar not in data_set:
                data_dict[tar] = []
                data_set.append(tar)
            data_dict[tar].append(ind)
        pro.append([data_set, data_dict])
    d_list = list()
    dnum_list = list()

    for ind, val in enumerate(pro):
        if len(val[0]) != 1:
            d_list.append(ind)
            dnum_list.append(min(map(lambda t: len(t), val[1].values())))
    return pro, d_list, dnum_list


if __name__ == "__main__":
    # dnum_list 분석
    d_list, dnum_list = process()[1:]
    print(d_list)  # 다른 경우를 가질 때의 아미노산 서열 + Python 순서라는 것에 주의!
    print(dnum_list)  # 이들이 몇개의 경우에서 mutaion이 검출되었는지 계산 + 순서는 위와 같음
    print(len(d_list))
    ans = list()
    for i in range(1, max(dnum_list) + 1):
        temp = list(filter(lambda t: t > i, dnum_list))
        ans.append(len(temp))
    print(ans)

    xti = [l + 1 for l in range(max(dnum_list))]
    rc('font', family="NanumGothic")
    fig = plt.figure(figsize=(9.60, 7.20))
    plt.plot(xti, ans)
    plt.xlabel("이상 서열 개수")
    plt.ylabel("아미노산 위치의 개수")
    plt.title("Mutaion 위치 그래프")
    plt.xticks(xti)

    plt.show()
