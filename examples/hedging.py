from dataclasses import replace
from environments.hedge import hedge_environment, State, Action
from tabular import monte_carlo
import tabular.encoding as encoding
import random
from environments.types import episode_fn
from pprint import pprint
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


n_state_bins = 51
n_action_bins = 51


def encode(state: State) -> int:
    encoded_base_price = encoding.encode_bucketed_range(
        0, 200, n_state_bins, state.base_price
    )
    return encoded_base_price


def decoder(action: int) -> Action:
    decoded = encoding.decode_bucketed_range(-100, 100, n_action_bins, action)
    return Action(order=decoded)


bucket_hedge = encoding.TabularEncoding(
    n_states=n_state_bins, encoder=encode, n_actions=n_action_bins, decoder=decoder
)

result = monte_carlo.monte_carlo_2(
    hedge_environment, random.Random(), bucket_hedge, 100000, 0.00005, 0.00005
)

rnd = random.Random()
pprint(result.params)
for x in np.linspace(0, 200, n_state_bins):
    hedge = result.policy(
        rnd, replace(hedge_environment.initial_state(rnd), base_price=x)
    )
    print(f"base {x} hedge {hedge.order}")
pprint(result.value_fn(rnd, hedge_environment.initial_state(rnd)))
ep = episode_fn(rnd, hedge_environment, result.policy)
total_return = sum(exp.reward for exp in ep)
print(f"total return: {total_return}")
print(f"final base price: {ep[-1].state.base_price}")
print(f"final hedge pos: {ep[-1].state.hedge_pos}")
pprint(ep)
