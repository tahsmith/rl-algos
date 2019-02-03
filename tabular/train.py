import numpy as np

from math_utils import moving_avg

def train(env, episode_fn, stopping_fn, logging_fn, steps_per_log, Q):
    returns = []
    tracks = []
    i = 0
    while not stopping_fn(returns, i, tracks):
        return_ = 0.0

        def step(action):
            nonlocal return_
            exp = env.step(action)
            return_ += exp[1]
            return exp

        Q_new = episode_fn(env.reset(), step, i, Q.copy())
        tracks.append(Q_new)
        returns.append(return_)
        Q = Q_new

        i += 1

        if i % steps_per_log == 0:
            logging_fn(returns, i, tracks)

    return returns, tracks


def train_with_plots(env, episode_fn, Q_initial, max_eps, steps_per_log,
                     value_range):
    import matplotlib.pyplot as plt

    def stop(returns, i, Q):
        return i == max_eps

    def log(returns, i, Q):
        avg = sum(returns[-steps_per_log:]) / steps_per_log
        print(f'\r{i:5d}: {avg}', end='')

    returns, tracks = train(env, episode_fn, stop, log, steps_per_log,
                            Q_initial)
    returns = np.array(returns)
    tracks = np.array(tracks)

    fig = plt.figure()
    plt.ylim(value_range)
    plt.xlabel('episode no.')
    plt.ylabel('return')
    plt.plot(returns, 'b.', label='return')
    avg_x, avg = moving_avg(steps_per_log, returns)
    plt.plot(avg_x, avg, 'r-', label=f'{steps_per_log}-long moving average')
    plt.legend()
    fig = plt.figure()
    plt.plot(np.max(tracks, axis=2))
    plt.xlabel('episode no.')
    plt.ylabel('state value')

    return returns, tracks