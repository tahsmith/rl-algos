from hypothesis import strategies as st, given

from tabular.encoding import (
    encode_bucketed_range,
    decode_bucketed_range,
    joint_encode,
    joint_decode,
)
from functools import partial


@given(st.integers(min_value=2, max_value=10))
def test_round_trip_exact_range(n_range: int):
    int_range = list(range(n_range))
    encoded_range = [
        encode_bucketed_range(0, n_range - 1, n_range, x) for x in int_range
    ]
    assert encoded_range == int_range
    decoded_range = [
        decode_bucketed_range(0, n_range - 1, n_range, x) for x in encoded_range
    ]
    assert decoded_range == int_range


def test_joint_encoding():
    def ident(x: int) -> int:
        return x

    joint_encoder = partial(joint_encode, ident, ident, 10)

    encoded = joint_encoder((9, 9))

    assert encoded == 99

    joint_decoder = partial(joint_decode, ident, ident, 10)

    decoded = joint_decoder(99)

    assert decoded == (9, 9)
