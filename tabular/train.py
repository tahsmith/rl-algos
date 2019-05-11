from copy import copy

import numpy as np

from math_utils import moving_avg


def train(env, episode_fn, stopping_fn, logging_fn, steps_per_log, *args):
    returns = []
    tracks = [[] for _ in args]
    i = 0
    while not stopping_fn(returns, i, *tracks):
        return_ = 0.0

        def step(action):
            nonlocal return_
            exp = env.step(action)
            return_ += exp[1]
            return exp

        args_new = episode_fn(env.reset(), step, i,
                              *[copy(arg) for arg in args])
        if not isinstance(args_new, tuple):
            args_new = (args_new,)
        for arg_new, track in zip(args_new, tracks):
            track.append(arg_new)
        returns.append(return_)
        args = args_new

        i += 1

        if i % steps_per_log == 0:
            logging_fn(returns, i, *tracks)

    return returns, tracks


def train_with_plots(fig, env, episode_fn, max_eps, steps_per_log,
                     value_range,
                     *args):
    def stop(returns, i, *args):
        return i == max_eps

    def log(returns, i, *args):
        avg = sum(returns[-steps_per_log:]) / steps_per_log
        print(f'\r{i:5d}: {avg}', end='')

    returns, tracks = train(env, episode_fn, stop, log, steps_per_log,
                             *args)
    returns = np.array(returns)
    q_track = np.array(tracks[0])

    axes = fig.subplots(1, 2)
    axes[0].set_ylim(value_range)
    axes[0].set_xlabel('episode no.')
    axes[0].set_ylabel('return')
    axes[0].plot(returns, 'b.', label='return')

    avg_x, avg = moving_avg(steps_per_log, returns)
    axes[0].plot(avg_x, avg, 'r-', label=f'{steps_per_log}-long moving average')
    axes[0].legend()

    axes[1].plot(np.max(q_track, axis=2))
    axes[1].set_xlabel('episode no.')
    axes[1].set_ylabel('state value')

    return returns, q_track
