import matplotlib.pyplot as plt
from matplotlib import rc


def process():
    """
    process는 아래와 같이 3개의 tuple를 반환한다
    (pro, d_list, dnum_list)
    이들은 각각 아래와 같은 의미를 가진다

    pro : 순서별로 몇개의 아미노산 종류가 있는지, 그것이 어느 것에는 어떤 것이 있는지 확인 ([[["M", "K"], {각각에 대한 딕셔너리}].. ]) + 또한 대상은 중복되지 않는다 (set과
    같은 list라 생각하자) d_list : 하나의 위치에 2개 이상의 아미노산을 가지는 대상들의 번호 dnum_list : d_list에서 가장 적은 개수를 가지는 아미노산의 개수 (몇개의 대상에서
    mutaion이 일어났는지 확인)
    """
    from datas import data_list
    data = data_list()
    data = list(map(lambda t: t[0], data))  # 단순 단백질 서열 list로 변경
    len_data = len(data[0])  # data의 모든 길이는 같다
    pro = list()
    for q in range(len_data):
        data_set = list()
        data_dict = dict()
        for ind, val in enumerate(data):
            tar = str(val[q])
            if tar not in data_set:
                data_dict[tar] = []
                data_set.append(tar)
            data_dict[tar].append(ind)
        pro.append([data_set, data_dict])
    dt_list = list()
    dtnum_list = list()

    for ind, val in enumerate(pro):
        if len(val[0]) != 1:
            dt_list.append(ind)
            dtnum_list.append(len(data) - max(map(lambda t: len(t), val[1].values())))  # 5/12 수정 (최소가 아닌 최대의 나머지 개수)
    return pro, dt_list, dtnum_list


if __name__ == "__main__":
    # dnum_list 분석하는 코드
    d_list, dnum_list = process()[1:]
    print(d_list)  # 다른 경우를 가질 때의 아미노산 서열 + Python 순서라는 것에 주의!
    print(dnum_list)  # 이들이 몇개의 경우에서 mutaion이 검출되었는지 계산 + 순서는 위와 같음
    print(len(d_list))
    ans = list()
    for i in range(1, max(dnum_list) + 1):
        temp = list(filter(lambda t: t > i, dnum_list))
        ans.append(len(temp))
    print(ans)

    xti = [ti + 1 for ti in range(max(dnum_list))]
    rc('font', family="NanumGothic")
    plt.figure(1, figsize=(9.60, 7.20))
    plt.plot(xti, ans)
    plt.xlabel("다른 서열을 가지는 Sample의 최소 개수")
    plt.ylabel("아미노산 위치의 개수")
    plt.title("Mutaion 위치와 최소 개수에 대한 그래프")
    plt.xticks(xti, [ti + 1 if not ti % 5 else "" for ti in range(max(dnum_list))])

    plt.figure(2)
    plt.hist(dnum_list, rwidth=0.9)
    plt.xlim([1, 51])
    plt.ylabel("아미노산 위치의 개수")
    plt.title("Mutaion 위치의 Sample 개수 분포")
    plt.show()
