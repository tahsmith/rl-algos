from environments.hedge import hedge_environment, State, Action
from tabular import monte_carlo
import tabular.encoding as encoding
import random
from environments.types import episode_fn
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO)

def encode(state: State) -> int:
    encoded = encoding.encode_bucketed_range(-100, 100, 5, state.hedge_pos)

    return encoded


def decoder(action: int) -> Action:
    decoded = encoding.decode_bucketed_range(-100, 100, 5, action)
    return Action(order=decoded)


bucket_hedge = encoding.TabularEncoding(
    n_states=5, encoder=encode, n_actions=5, decoder=decoder
)

result = monte_carlo.monte_carlo_2(
    hedge_environment, random.Random(), bucket_hedge, 200000, 0.001, 0.0001 
)

rnd = random.Random()
pprint(result.params)
pprint(result.policy(rnd, hedge_environment.initial_state(rnd)))
pprint(result.value_fn(rnd, hedge_environment.initial_state(rnd)))
ep = episode_fn(rnd, hedge_environment, result.policy)
total_return = sum(exp.reward for exp in ep)
print(f"total return: {total_return}" )
print(f"final base price: {ep[-1].state.base_price}")
print(f"final hedge pos: {ep[-1].state.hedge_pos}")

