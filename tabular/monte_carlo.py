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

from environments.types import Step, StepFn, History, Environment, episode_fn
from .types import TabularAction, TabularState, Array, ActionDecoder, StateEncoder


def epsilon_greedy_policy_probs(Q: Array, eps: float, state: int):
    action_size = len(Q[0])
    probs = np.ones(action_size) * eps / action_size
    probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
    return probs


def epsilon_greedy(random: Random, eps: float, Q: Array, state: int) -> TabularAction:
    return np.random.choice(len(Q[state]), p=epsilon_greedy_policy_probs(Q, eps, state))


type EncodedState[State] = tuple[State, TabularState]
type EncodedAction[Action] = tuple[Action, TabularAction]


def tabular_epsilon_greedy[State, Action](
    eps: float,
    decoder: ActionDecoder[Action],
    q: Array,
    random: Random,
    state: EncodedState[State],
) -> EncodedAction[Action]:
    encoded_action = epsilon_greedy(random, eps, q, state[1])
    return (decoder(encoded_action), encoded_action)


def tabular_step_fn[State, Action](
    step_fn: StepFn[State, Action],
    encode: StateEncoder[State],
    random: Random,
    state: EncodedState[State],
    action: EncodedAction[Action],
) -> Step[EncodedState[State]]:
    step = step_fn(random, state[0], action[0])
    return Step(
        reward=step.reward,
        next_state=(step.next_state, encode(step.next_state))
        if step.next_state is not None
        else None,
    )


@dataclass
class TabularEncoding[State, Action]:
    n_states: int
    n_actions: int
    encoder: StateEncoder[State]
    decoder: ActionDecoder[Action]


type TabularPolicy[State, Action] = Callable[[Array, Random, State], Action]
type TabularLearner[State, Action] = Callable[
    [Environment[State, Action], TabularEncoding[State, Action]],
    tuple[Array, TabularPolicy[State, Action]],
]


def encode_environment[State, Action](
    environment: Environment[State, Action], encoding: TabularEncoding[State, Action]
) -> Environment[EncodedState[State], EncodedAction[Action]]:
    return Environment[EncodedState[State], EncodedAction[Action]](
        initial_state=lambda random: (
            state := environment.initial_state(random),
            encoding.encoder(state),
        ),
        step_fn=lambda random, state, action: Step(
            reward=(step := environment.step_fn(random, state[0], action[0])).reward,
            next_state=(step.next_state, encoding.encoder(step.next_state))
            if step.next_state is not None
            else None,
        ),
    )


def monte_carlo_2[State, Action](
    environment: Environment[State, Action],
    random: Random,
    trials: int,
    encoding: TabularEncoding[State, Action]
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
