import gym

from math_utils import constant
from tabular import train


def example_episode(state, step, i, a1, a2):
    return a1, a2


def example_stopping(returns, i, a1_track, a2_track):
    return i == 100


def test_train():
    returns, tracks = train(gym.make('FrozenLake-v0'), example_episode,
                   example_stopping, constant(None), 1, 'x', 'y')
    assert len(tracks) == 2
    assert len(tracks[0]) == 100
    assert all(x == 'x' for x in tracks[0])
    assert all(y == 'y' for y in tracks[1])
