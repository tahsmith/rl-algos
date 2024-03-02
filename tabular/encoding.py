from .types import (
    EncodedAction,
    EncodedState,
    Array,
    TabularAction,
    StateEncoder,
    ActionDecoder,
)
from random import Random

from environments.types import StepFn, Environment, Step
from dataclasses import dataclass
import numpy as np
from typing import Callable, TypeVar, Generic, TypeVarTuple, Unpack


def epsilon_greedy_policy_probs(Q: Array, eps: float, state: int):
    action_size = len(Q[0])
    probs = np.ones(action_size) * eps / action_size
    probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
    return probs


def epsilon_greedy(random: Random, eps: float, Q: Array, state: int) -> TabularAction:
    return np.random.choice(len(Q[state]), p=epsilon_greedy_policy_probs(Q, eps, state))


State = TypeVar("State")
Action = TypeVar("Action")


def tabular_epsilon_greedy(
    eps: float,
    decoder: ActionDecoder[Action],
    q: Array,
    random: Random,
    state: EncodedState[State],
) -> EncodedAction[Action]:
    encoded_action = epsilon_greedy(random, eps, q, state[1])
    return (decoder(encoded_action), encoded_action)


def tabular_step_fn(
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
class TabularEncoding(Generic[State, Action]):
    n_states: int
    n_actions: int
    encoder: StateEncoder[State]
    decoder: ActionDecoder[Action]


TabularPolicy = Callable[[Array, Random, State], Action]
TabularLearner = Callable[
    [Environment[State, Action], TabularEncoding[State, Action]],
    tuple[Array, TabularPolicy[State, Action]],
]


def encode_environment(
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


def encode_bucketed_range(
    min_value: float, max_value: float, n_levels: int, value: float
) -> int:
    value = (value - min_value) / (max_value - min_value) * (n_levels - 1)
    if value >= n_levels:
        return n_levels - 1
    if value < 0:
        return 0
    return int(value)


def decode_bucketed_range(
    min_value: float, max_value: float, n_levels: int, value: int
) -> float:
    return value / (n_levels - 1) * (max_value - min_value) + min_value


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Ts = TypeVarTuple("Ts")

Encoder = Callable[[T], int]


def joint_encode(
    encoder1: Encoder[T1],
    encoder2: Encoder[T2],
    n_states2: int,
    value: tuple[T1, T2],
) -> int:
    x1, x2 = value
    encoded1 = encoder1(x1)
    encoded2 = encoder2(x2)

    return encoded1 + encoded2 * n_states2


def append_encoding(
    encoder1: Encoder[tuple[Unpack[Ts]]],
    encoder: Encoder[T],
    n_states: int,
    value: tuple[Unpack[Ts], T],
) -> int:
    *xs, x = value
    reveal_type(value[:-1])
    tup: tuple[Unpack[Ts]] = tuple(xs)
    return joint_encode(encoder1, encoder, n_states, (tup, x))


Decoder = Callable[[int], T]


def joint_decode(
    decoder1: Decoder[T1], decoder2: Decoder[T2], n_states2: int, value: int
) -> tuple[T1, T2]:
    encoded1 = value % n_states2
    encoded2 = value // n_states2

    return (decoder1(encoded1), decoder2(encoded2))
