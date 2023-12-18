import numpy as np
from typing import Callable
from numpy.typing import NDArray

type TabularState = int
type TabularAction = int
type Array = NDArray[np.float64]
type StateEncoder[TState] = Callable[[TState], TabularState]
type ActionDecoder[TAction] = Callable[[TabularAction], TAction]

type LearningState[TState, TArray] = tuple[TState, TArray, int]
type LearningSchedule = Callable[[int], float]
