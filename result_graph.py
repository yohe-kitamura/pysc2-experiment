import numpy as np
import matplotlib.pyplot as plt
import time

calculation_unit = 50
WIN = "1"
LOOSE = "-1"

while True:

    print("start")

    results = []
    win_dicts = {}
    loose_dicts = {}
    with open('result.txt', 'r') as f:
        for row in f:
            results.append(row.strip())

    win_count = 0
    loose_count = 0
    for i, result in enumerate(results):
        if result == WIN:
            win_count += 1
        elif result == LOOSE:
            loose_count += 1

        # 一定回数毎に集計
        if (i + 1) % calculation_unit is 0:
            win_dicts[str(int((i + 1) / calculation_unit))] = win_count / calculation_unit
            loose_dicts[str(int((i + 1) / calculation_unit))] = loose_count / calculation_unit
            win_count = 0
            loose_count = 0

    plt.plot(np.array(list(win_dicts.keys())), np.array(list(win_dicts.values())), color="red")
    plt.xlabel("number of trials")
    plt.ylabel("percentage of victories(%)")

    plt.plot(np.array(list(loose_dicts.keys())), np.array(list(loose_dicts.values())), color="blue")

    plt.show()

    time.sleep(60)

    print("close")
    plt.close()
