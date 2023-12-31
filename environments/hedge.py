from dataclasses import dataclass, replace
from random import Random
import math
from .types import Step, Environment


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


def step_fn(rng: Random, state: State, action: Action) -> Step[State]:
    next_state = replace(
        state,
        tte=state.tte - 1,
        base_price=state.base_price
        * math.exp(rng.normalvariate(sigma=state.volatility)),
        hedge_pos=action.order + state.hedge_pos,
    )

    if state.tte == 1:
        expiry_value = (
            max(0, next_state.base_price - state.strike) * 100
            + next_state.hedge_pos * next_state.base_price
        )
    else:
        expiry_value = 0

    done = next_state.tte == 0

    step = Step(
        next_state=next_state if not done else None,
        reward=-action.order * state.base_price + expiry_value,
    )

    return step


hedge_environment = Environment(
    initial_state=lambda _: State(
        strike=100, tte=100, base_price=100, hedge_pos=0, volatility=1.0
    ),
    step_fn=step_fn,
)
