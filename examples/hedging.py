from environments.hedge import hedge_environment, State, Action
from tabular import monte_carlo
import tabular.encoding as encoding
import random
from environments.types import episode_fn
from functools import partial


def encode(state: State) -> int:
    encoded = encoding.encode_bucketed_range(-100, 100, 5, state.hedge_pos)

    return encoded


def decoder(action: int) -> Action:
    decoded = encoding.decode_bucketed_range(-100, 100, 5, action)
    return Action(order=decoded)


bucket_hedge = encoding.TabularEncoding(
    n_states=5, encoder=encode, n_actions=5, decoder=decoder
)

q, policy = monte_carlo.monte_carlo_2(
    hedge_environment, random.Random(), 100000, bucket_hedge
)

rnd = random.Random()
print(q)
print(policy(q, rnd, hedge_environment.initial_state(rnd)))
ep = episode_fn(rnd, hedge_environment, partial(policy, q))
__import__('pprint').pprint(ep)
