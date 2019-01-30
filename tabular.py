import numpy as np
from matplotlib import pyplot as plt


def train(episode_fn, Q_initial):
    state_size = Q_initial.shape[0]
    action_size = Q_initial.shape[1]
    N = 100
    M = int(5e3)
    returns = [float('inf') for _ in range(N)]
    Q = Q_initial
    Q_track = np.zeros((M, state_size, action_size), np.float64)
    for i in range(int(M)):
        return_, Q_new = episode_fn(i, Q.copy())
        Q_track[i, :, :] = Q_new
        returns = [return_, *returns[0:-1]]
        diff = np.abs(Q_new - Q).mean()
        Q = Q_new
        if (i + 1) % N == 0:
            score = sum(returns) / N
            print(diff, score, returns[-1])

            if 1.0 < score:
                break

    V = np.max(Q_track, 2)
    # print(np.max(Q_track[-1, :, :], 1).reshape(4, 12))

    plt.plot(V)
    plt.xscale('log')
    plt.show()
