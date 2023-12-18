from dataclasses import dataclass
from typing import Callable, Optional
from random import Random


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

type Policy[TState, TAction] = Callable[[Random, TState], TAction]

type History[Action, State] = list[Experience[State, Action]]


@dataclass
class Environment[State, Action]:
    initial_state: Callable[[Random], State]
    step_fn: StepFn[State, Action]


def episode_fn[State, Action](
    random: Random,
    environment: Environment[State, Action],
    policy: Policy[State, Action],
    max_steps: Optional[int] = None,
) -> History[Action, State]:
    state = environment.initial_state(random)
    step_fn = environment.step_fn
    history: History[Action, State] = []
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
