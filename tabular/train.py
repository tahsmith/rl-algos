import numpy as np


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
            logging_fn(returns, i,  tracks)


    return returns, tracks
