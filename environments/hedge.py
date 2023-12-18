from pprint import pprint
from dataclasses import dataclass, replace
from typing import Callable
from random import Random
import math


@dataclass
class State:
    strike: float
    tte: float
    base_price: float
    hedge_pos: float
    volatility: float


@dataclass
class Action:
    order: float


def step_fn(rng: Random, state: State, action: Action) -> Step:
    if state.tte == 1:
        expiry_value = max(0, state.base_price - state.strike)
    else:
        expiry_value = 0
    next_state = replace(
        state,
        tte=state.tte - 1,
        base_price=state.base_price
        * math.exp(rng.normalvariate(sigma=state.volatility)),
        hedge_pos=action.order + state.hedge_pos,
    )

    step = Step(
        state=next_state, reward=-action.order + expiry_value, done=next_state.tte == 0
    )

    return step


def random_action(rng: Random, state: State) -> Action:
    return Action(order=rng.choice([-1, 1]))


def episode_fn(
    random: Random, initial_state: State, step_fn: StepFn, action_fn: ActionFn
):
    state = initial_state
    done = False
    history = []
    while not done:
        action = action_fn(random, state)
        step = step_fn(random, state, action)
        history.append((action, step))
        done = step.done
        state = step.state
    return history


steps = episode_fn(Random(1), State(100, 10, 100, 0, 1.0), step_fn, random_action)
pprint(steps)
