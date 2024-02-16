from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar
from random import Random


State = TypeVar("State")
Action = TypeVar("Action")


@dataclass
class Step(Generic[State]):
    reward: float
    next_state: Optional[State]


@dataclass
class Experience(Generic[State, Action]):
    state: State
    action: Action
    reward: float
    next_state: Optional[State]


StepFn = Callable[[Random, State, Action], Step[State]]

Policy = Callable[[Random, State], Action]

History = list[Experience[State, Action]]


@dataclass
class Environment(Generic[State, Action]):
    initial_state: Callable[[Random], State]
    step_fn: StepFn[State, Action]


def episode_fn(
    random: Random,
    environment: Environment[State, Action],
    policy: Policy[State, Action],
    max_steps: Optional[int] = None,
) -> History[State, Action]:
    state = environment.initial_state(random)
    step_fn = environment.step_fn
    history: History[State, Action] = []
    i = 0
    while state is not None:
        if max_steps and i > max_steps:
            break
        action = policy(random, state)
        step = step_fn(random, state, action)
        experience = Experience(
            state=state, action=action, reward=step.reward, next_state=step.next_state
        )
        history.append(experience)
        state = step.next_state
        i += 1
    return history
