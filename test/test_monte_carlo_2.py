from random import Random
from typing import Literal
from tabular.monte_carlo import monte_carlo_2, Step

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


type GridCellType = Literal["s", "g", "h", "."]
type GridAction = Literal["n", "s", "e", "w"]
type GridState = tuple[int, int]
@dataclass
class GridWorld:
    columns: int
    rows: int
    grid: list[list[GridCellType]]

def grid
