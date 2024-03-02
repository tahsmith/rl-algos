from functools import partial
from random import Random
from typing import Literal, TypeAlias, get_args

from hypothesis.strategies._internal.core import DataStrategy
from environments.types import Step
from tabular.monte_carlo import (
    Environment,
    TabularEncoding,
    episode_fn,
    monte_carlo_2,
)
from dataclasses import dataclass
from hypothesis import given, note, strategies as st
import numpy as np

CorridorState: TypeAlias = Literal["h", "s", "g"]
CorridorAction: TypeAlias = Literal["l", "r"]


def corridor_step_fn(
    random: Random, state: CorridorState, action: CorridorAction
) -> Step[CorridorState]:
    if state == "s":
        if action == "l":
            return Step(reward=0, next_state=None)
        if action == "r":
            return Step(reward=1, next_state=None)
    else:
        raise ValueError()


corridor_environment = Environment(
    step_fn=corridor_step_fn, initial_state=lambda _: "s"
)


def encodeState(state: CorridorState) -> int:
    return get_args(CorridorState).index(state)


def decodeAction(action: int) -> CorridorAction:
    return get_args(CorridorAction)[action]


encoding = TabularEncoding(3, 2, encodeState, decodeAction)


def test_monte_carlo_corridor():
    q = monte_carlo_2(corridor_environment, Random(), encoding, 100, 0.001, 0.001)
    print(q)


GridCellType: TypeAlias = Literal["g", "h", "."]
GridAction: TypeAlias = Literal["n", "s", "e", "w"]
GridState: TypeAlias = tuple[int, int]


@dataclass
class GridWorld:
    columns: int
    rows: int
    grid: list[list[GridCellType]]


def grid_world_step_fn(
    world: GridWorld, random: Random, state: GridState, action: GridAction
) -> Step[GridState]:
    if action == "n":
        state = (state[0] - 1, state[1])
    elif action == "s":
        state = (state[0] + 1, state[1])
    elif action == "e":
        state = (state[0], state[1] + 1)
    elif action == "w":
        state = (state[0], state[1] - 1)
    else:
        raise ValueError()

    if (
        (state[0] > world.rows - 1)
        or (state[0] < 0)
        or (state[1] > world.columns - 1)
        or (state[1] < 0)
    ):
        return Step(reward=0, next_state=None)

    cell_type = world.grid[state[0]][state[1]]

    if cell_type == ".":
        return Step(reward=0, next_state=state)
    elif cell_type == "h":
        return Step(reward=0, next_state=None)
    elif cell_type == "g":
        return Step(reward=1, next_state=None)
    else:
        raise ValueError()


def make_grid_world(world: GridWorld) -> Environment[GridState, GridAction]:
    return Environment(
        lambda _: (0, 0),
        partial(grid_world_step_fn, world),
    )


@given(st.data())
def test_grid_world(data: DataStrategy):
    world = GridWorld(rows=2, columns=2, grid=[[".", "h"], [".", "g"]])
    environmnet = make_grid_world(world)

    encoding = TabularEncoding[GridState, GridAction](
        4,
        4,
        lambda state: state[0] * 2 + state[1],
        lambda action: get_args(GridAction)[action],
    )

    result = monte_carlo_2(environmnet, Random(), encoding, 1000, 0.001, 0.001)

    q = data.draw(
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False), max_size=4, min_size=4
            ),
            max_size=4,
            min_size=4,
        )
    )
    q = np.array(q, dtype=np.float64)

    history_star = episode_fn(
        Random(),
        environmnet,
        result.policy,
        100,
    )
    history = episode_fn(
        Random(), environmnet, partial(result.parametric_policy, q), 100
    )

    return_ = sum(x.reward for x in history)
    return_star = sum(x.reward for x in history_star)

    note(str(result.params))
    note(str(history_star))
    note(str(q))
    note(str(history))

    assert return_star >= return_
