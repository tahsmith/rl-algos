from functools import partial
from random import Random
from typing import Literal, TypeAlias, get_args

from hypothesis.strategies._internal.core import DataStrategy
from tabular.monte_carlo import episode_fn, monte_carlo_2, Step
from dataclasses import dataclass
from hypothesis import given, note, strategies as st
import numpy as np

type CorridorState = Literal["h", "s", "g"]
type CorridorAction = Literal["l", "r"]


def corridor_step_fn(
    random: Random, state: CorridorState, action: CorridorAction
) -> Step[CorridorState]:
    if state == "s":
        if action == "l":
            return Step(state="h", reward=0, done=True)
        if action == "r":
            return Step(state="g", reward=1, done=True)
    else:
        raise ValueError()


def test_monte_carlo_corridor():
    q = monte_carlo_2(
        Random(),
        "s",
        100,
        corridor_step_fn,
        3,
        2,
        lambda s: ["h", "s", "g"].index(s),
        lambda a: ["l", "r"][a],
    )
    print(q)


type GridCellType = Literal["g", "h", "."]
GridAction: TypeAlias = Literal["n", "s", "e", "w"]
type GridState = tuple[int, int]


@dataclass
class GridWorld:
    columns: int
    rows: int
    grid: list[list[GridCellType]]


def grid_world(
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
        return Step(state=state, reward=0, done=True)

    cell_type = world.grid[state[0]][state[1]]

    if cell_type == ".":
        return Step(state=state, reward=0, done=False)
    elif cell_type == "h":
        return Step(state=state, reward=0, done=True)
    elif cell_type == "g":
        return Step(state=state, reward=1, done=True)
    else:
        raise ValueError()


@given(st.data())
def test_grid_world(data: DataStrategy):
    world = GridWorld(rows=2, columns=2, grid=[[".", "h"], [".", "g"]])

    step_fn = partial(grid_world, world)

    q_star, policy = monte_carlo_2(
        Random(),
        (0, 0),
        100,
        step_fn,
        4,
        4,
        lambda state: state[0] * 2 + state[1],
        lambda action: get_args(GridAction)[action],
    )

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
        (0, 0),
        step_fn,
        partial(policy, q_star),
        100,
    )
    history = episode_fn(Random(), (0, 0), step_fn, partial(policy, q), 100)

    return_ = sum(x[1].reward for x in history)
    return_star = sum(x[1].reward for x in history_star)

    note(str(q_star))
    note(str(history_star))
    note(str(q))
    note(str(history))

    assert return_star >= return_
