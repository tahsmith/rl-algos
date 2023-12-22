"""
This method of learning the state value function v(s) is basically straight
from the definition of v(s) as the expected reward following the policy from
s to the end of the episode. We use random simulation to collect samples to
estimate the expectation value.
"""


from typing import Callable
from random import Random
import numpy as np
from functools import partial

from environments.types import History, Environment, episode_fn
from .types import (
    EncodedState,
    EncodedAction,
    Array,
)
from .encoding import TabularEncoding, encode_environment, tabular_epsilon_greedy


def monte_carlo_2[State, Action](
    environment: Environment[State, Action],
    random: Random,
    trials: int,
    encoding: TabularEncoding[State, Action],
) -> tuple[Array, Callable[[Array, Random, State], Action]]:
    q: Array = np.zeros((encoding.n_states, encoding.n_actions))
    encoded_environment = encode_environment(environment, encoding)
    for i in range(trials):
        eps = 1 / (i * 0.01 + 1)
        policy = partial(tabular_epsilon_greedy, eps, encoding.decoder, q)
        history = episode_fn(random, encoded_environment, policy)
        alpha = 1 / (i * 0.01 + 1)
        q = monte_carlo_update(alpha, q, history)

    return (
        q,
        lambda q, random, state: tabular_epsilon_greedy(
            0.0, encoding.decoder, q, random, (state, encoding.encoder(state))
        )[0],
    )


def monte_carlo_update[State, Action](
    alpha: float,
    q: Array,
    history: History[EncodedState[State], EncodedAction[Action]],
) -> Array:
    q = q.copy()
    states = [x.state[1] for x in history]
    actions = [x.action[1] for x in history]
    rewards = [x.reward for x in history]
    future_rewards = np.cumsum(rewards[::-1])[::-1]
    for state, action, future_reward in zip(states, actions, future_rewards):
        q[state, action] += alpha * (future_reward - q[state, action])

    return q
