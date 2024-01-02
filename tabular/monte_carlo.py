"""
This method of learning the state value function v(s) is basically straight
from the definition of v(s) as the expected reward following the policy from
s to the end of the episode. We use random simulation to collect samples to
estimate the expectation value.
"""


from dataclasses import dataclass
from typing import Callable
from random import Random
import numpy as np
from functools import partial
from logging import getLogger

from environments.types import History, Environment, Policy, episode_fn
from .types import (
    EncodedState,
    EncodedAction,
    Array,
)
from .encoding import TabularEncoding, encode_environment, tabular_epsilon_greedy


type ParametricPolicy[Params, State, Action] = Callable[[Params, Random, State], Action]
type ParametricStateValueFn[Params, State] = Callable[[Params, Random, State], float]
type StateValueFn[State] = Callable[[Random, State], float]

logger = getLogger(__name__)


@dataclass
class EstimationResult[Params, State, Action]:
    params: Params
    parametric_policy: ParametricPolicy[Params, State, Action]
    parametric_state_value_fn: ParametricStateValueFn[Params, State]

    @property
    def policy(self) -> Policy[State, Action]:
        return partial(self.parametric_policy, self.params)

    @property
    def value_fn(self) -> StateValueFn[State]:
        return partial(self.parametric_state_value_fn, self.params)


def monte_carlo_2[State, Action](
    environment: Environment[State, Action],
    random: Random,
    encoding: TabularEncoding[State, Action],
    trials: int,
    epsilon_decay_rate: float,
    alpha_decay_rate: float
) -> EstimationResult[Array, State, Action]:
    q: Array = np.zeros((encoding.n_states, encoding.n_actions))
    encoded_environment = encode_environment(environment, encoding)
    for i in range(trials):
        eps = 1 / (i * epsilon_decay_rate + 1)
        policy = partial(tabular_epsilon_greedy, eps, encoding.decoder, q)
        history = episode_fn(random, encoded_environment, policy)
        alpha = 1 / (i * alpha_decay_rate + 1)
        q = monte_carlo_update(alpha, q, history)
        logger.info(
            f"trial {i} of {trials}: eps={eps:.5} alpha={alpha:.5} reward={sum(exp.reward for exp in history):.5}"
        )

    return EstimationResult(
        q,
        lambda q, random, state: tabular_epsilon_greedy(
            0.0, encoding.decoder, q, random, (state, encoding.encoder(state))
        )[0],
        lambda q, _, state: q[encoding.encoder(state), :].max(),
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
    visited = np.zeros_like(q, dtype=np.bool_)
    for state, action, future_reward in zip(states, actions, future_rewards):
        if not visited[state, action]:
            q[state, action] += (future_reward - q[state, action]) * alpha
            visited[state, action] = True

    return q
