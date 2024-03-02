import numpy as np
from typing import Callable, TypeVar
from numpy.typing import NDArray

TState = TypeVar("TState")
TAction = TypeVar("TAction")
TArray = TypeVar("TArray")

TabularState = int
TabularAction = int
Array = NDArray[np.float64]
StateEncoder = Callable[[TState], TabularState]
ActionDecoder = Callable[[TabularAction], TAction]

LearningState = tuple[TState, TArray, int]
LearningSchedule = Callable[[int], float]

EncodedState = tuple[TState, TabularState]
EncodedAction = tuple[TAction, TabularAction]
