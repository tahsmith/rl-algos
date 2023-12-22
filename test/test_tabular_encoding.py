from hypothesis import strategies as st, given

from tabular.encoding import encode_bucketed_range, decode_bucketed_range

@given(st.integers(min_value=2, max_value=10))
def test_round_trip_exact_range(n_range: int):
    int_range = list(range(n_range))
    encoded_range = [encode_bucketed_range(0, n_range - 1, n_range, x) for x in int_range]
    assert encoded_range == int_range 
    decoded_range = [decode_bucketed_range(0, n_range - 1, n_range, x) for x in encoded_range]
    assert decoded_range == int_range

