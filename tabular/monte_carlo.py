"""
This method of learning the state value function v(s) is basically straight
from the definition of v(s) as the expected reward following the policy from
s to the end of the episode. We use random simulation to collect samples to
estimate the expectation value.
"""


def monte_carlo(policy, alpha):
    def episode(state, step, i, Q):
        done = False
        experiences = []
        while not done:
            action = policy(Q, i, state)
            next_state, reward, done, info = step(action)
            experiences.append((state, action, reward, next_state, done))
            state = next_state

        states, actions, rewards, _, __ = zip(*experiences)

        for j, state in enumerate(states):
            old_Q = Q[state][actions[j]]
            Q[state][actions[j]] = old_Q + alpha(i) * (sum(rewards[j:]) - old_Q)
        return Q

    return episode


from dataclasses import dataclass
from typing import Callable
from random import Random
import numpy as np
from functools import partial
from numpy.typing import NDArray


@dataclass
class Step[TState]:
    state: TState
    reward: float
    done: bool


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


def epsilon_greedy(eps: LearningSchedule) -> Callable[[Array, int, int], int]:
    def policy(Q: Array, i: int, state: int):
        return np.random.choice(
            len(Q[state]), p=epsilon_greedy_policy_probs(Q, eps(i), state)
        )

    return policy


type History[Action, State] = list[tuple[Action, Step[State]]]


def episode_fn[State, Action](
    random: Random,
    initial_state: State,
    step_fn: StepFn[State, Action],
    action_fn: ActionFn[State, Action],
) -> History[Action, State]:
    state = initial_state
    done = False
    history: History[Action, State] = []
    while not done:
        action = action_fn(random, state)
        step = step_fn(random, state, action)
        history.append((action, step))
        done = step.done
        state = step.state
    return history


def tabular_epsilon_greedy[State, Action](
    eps: float,
    decoder: ActionDecoder[Action],
    q: Array,
    random: Random,
    state: tuple[State, EncodedState],
) -> tuple[Action, EncodedAction]:
    i = 0
    encoded_action = epsilon_greedy(lambda _: eps)(q, i, state[1])
    return (decoder(encoded_action), encoded_action)


def tabular_step_fn[State, Action](
    step_fn: StepFn[State, Action],
    encode: StateEncoder[State],
    random: Random,
    state: tuple[State, EncodedState],
    action: tuple[Action, EncodedAction],
) -> Step[tuple[State, EncodedState]]:
    step = step_fn(random, state[0], action[0])
    return Step(
        state=(step.state, encode(step.state)), reward=step.reward, done=step.done
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
):
    q: Array = np.zeros((n_states, n_actions))
    encoded_step_fn = partial(tabular_step_fn, step_fn, encoder)
    encoded_initial_state = (initial_state, encoder(initial_state))
    for i in range(trials):
        eps = 1 / (i + 1)
        policy = partial(tabular_epsilon_greedy, eps, decoder, q)
        history = episode_fn(random, encoded_initial_state, encoded_step_fn, policy)
        alpha = 1 / (i + 1)
        q = monte_carlo_update(alpha, q, encoded_initial_state, history)

    return q


def monte_carlo_update[State, Action](
    alpha: float,
    q: Array,
    initial_state: tuple[State, EncodedState],
    history: History[tuple[Action, EncodedAction], tuple[State, EncodedState]],
) -> Array:
    q = q.copy()
    states = [initial_state[1], *(x[1].state[1] for x in history[:-1])]
    actions = [x[0][1] for x in history]
    rewards = [x[1].reward for x in history]
    future_rewards = np.cumsum(rewards[::-1])[::-1]
    for state, action, future_reward in zip(states, actions, future_rewards):
        q[state, action] += alpha * (future_reward - q[state, action])

    return q
