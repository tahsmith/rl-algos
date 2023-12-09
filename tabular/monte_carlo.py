"""
This method of learning the state value function v(s) is basically straight
from the definition of v(s) as the expected reward following the policy from
s to the end of the episode. We use random simulation to collect samples to
estimate the expectation value.
"""


from dataclasses import dataclass
from typing import Callable, Optional
from random import Random
import numpy as np
from functools import partial
from numpy.typing import NDArray


@dataclass
class Step[State]:
    reward: float
    next_state: Optional[State]


@dataclass
class Experience[State, Action]:
    state: State
    action: Action
    reward: float
    next_state: Optional[State]


type StepFn[TState, TAction] = Callable[[Random, TState, TAction], Step[TState]]
type ActionFn[TState, TAction] = Callable[[Random, TState], TAction]

type EncodedState = int
type EncodedAction = int
type Array = NDArray[np.float64]
type StateEncoder[TState] = Callable[[TState], EncodedState]
type ActionDecoder[TAction] = Callable[[EncodedAction], TAction]

type LearningState[TState, TArray] = tuple[TState, TArray, int]
type LearningSchedule = Callable[[int], float]


def epsilon_greedy_policy_probs(Q: Array, eps: float, state: int):
    action_size = len(Q[0])
    probs = np.ones(action_size) * eps / action_size
    probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
    return probs


def epsilon_greedy(random: Random, eps: float, Q: Array, state: int) -> EncodedAction:
    return np.random.choice(len(Q[state]), p=epsilon_greedy_policy_probs(Q, eps, state))


type History[Action, State] = list[Experience[State, Action]]


def episode_fn[State, Action](
    random: Random,
    initial_state: State,
    step_fn: StepFn[State, Action],
    action_fn: ActionFn[State, Action],
    max_steps: Optional[int] = None,
) -> History[Action, State]:
    state = initial_state
    history: History[Action, State] = []
    i = 0
    while state is not None:
        if max_steps and i > max_steps:
            break
        action = action_fn(random, state)
        step = step_fn(random, state, action)
        experience = Experience(
            state=state, action=action, reward=step.reward, next_state=step.next_state
        )
        history.append(experience)
        state = step.next_state
        i += 1
    return history


type AugmentedState[State] = tuple[State, EncodedState]
type AugmentedAction[Action] = tuple[Action, EncodedAction]


def tabular_epsilon_greedy[State, Action](
    eps: float,
    decoder: ActionDecoder[Action],
    q: Array,
    random: Random,
    state: AugmentedState[State],
) -> AugmentedAction[Action]:
    encoded_action = epsilon_greedy(random, eps, q, state[1])
    return (decoder(encoded_action), encoded_action)


def tabular_step_fn[State, Action](
    step_fn: StepFn[State, Action],
    encode: StateEncoder[State],
    random: Random,
    state: AugmentedState[State],
    action: AugmentedAction[Action],
) -> Step[AugmentedState[State]]:
    step = step_fn(random, state[0], action[0])
    return Step(
        reward=step.reward,
        next_state=(step.next_state, encode(step.next_state))
        if step.next_state is not None
        else None,
    )


def monte_carlo_2[State, Action](
    random: Random,
    initial_state: State,
    trials: int,
    step_fn: StepFn[State, Action],
    n_states: int,
    n_actions: int,
    encoder: StateEncoder[State],
    decoder: ActionDecoder[Action],
) -> tuple[Array, Callable[[Array, Random, State], Action]]:
    q: Array = np.zeros((n_states, n_actions))
    encoded_step_fn = partial(tabular_step_fn, step_fn, encoder)
    encoded_initial_state = (initial_state, encoder(initial_state))
    for i in range(trials):
        eps = 1 / (i * 0.01 + 1)
        policy = partial(tabular_epsilon_greedy, eps, decoder, q)
        history = episode_fn(random, encoded_initial_state, encoded_step_fn, policy)
        alpha = 1 / (i * 0.01 + 1)
        q = monte_carlo_update(alpha, q, history)

    return (
        q,
        lambda q, random, state: tabular_epsilon_greedy(
            0.0, decoder, q, random, (state, encoder(state))
        )[0],
    )


def monte_carlo_update[State, Action](
    alpha: float,
    q: Array,
    history: History[tuple[Action, EncodedAction], tuple[State, EncodedState]],
) -> Array:
    q = q.copy()
    states = [x.state[1] for x in history]
    actions = [x.action[1] for x in history]
    rewards = [x.reward for x in history]
    future_rewards = np.cumsum(rewards[::-1])[::-1]
    for state, action, future_reward in zip(states, actions, future_rewards):
        q[state, action] += alpha * (future_reward - q[state, action])

    return q
