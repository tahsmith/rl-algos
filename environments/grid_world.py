from functools import partial
from random import Random
from typing import Literal, TypeAlias

from environments.types import Step
from tabular.monte_carlo import (
    Environment,
)
from dataclasses import dataclass


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

