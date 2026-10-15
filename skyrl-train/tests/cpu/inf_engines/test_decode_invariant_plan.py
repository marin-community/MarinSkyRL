import pytest

from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import fa3_invariant_requests, fa3_request_calls


@pytest.mark.parametrize(
    "window, pieces",
    [
        # Full layer: the unaligned prefill's rows before position 8, then the rest; a chunk past position 8 lies
        # within one 4-position group, so it stays whole.
        (None, [(5, 3), (8, 3), (19, 1), (0, 5), (2, 1), (9, 3), (4, 3), (4, 6)]),
        # Sliding-window layer: rows at positions 8 and later are alone.
        (
            8,
            [
                (5, 3),
                (8, 1),
                (9, 1),
                (10, 1),
                (19, 1),
                (0, 5),
                (2, 1),
                (9, 1),
                (10, 1),
                (11, 1),
                (4, 3),
                (4, 4),
                (8, 1),
                (9, 1),
            ],
        ),
    ],
)
def test_fa3_invariant_requests_start_prefills_aligned_and_run_rows_past_the_window_alone(window, pieces):
    alignment = 4
    # (query rows, keys) per request: a prefill from position 5 across position 8, a decode row past it, a prefill
    # from position 0, a decode row, a prefill chunk from position 9, and prefills from aligned cached prefixes, the
    # last one across position 8.
    requests = [(6, 11), (1, 20), (5, 5), (1, 3), (3, 12), (3, 7), (6, 10)]
    query_start = [0]
    for rows, _ in requests:
        query_start.append(query_start[-1] + rows)

    split = fa3_invariant_requests(query_start, [keys for _, keys in requests], window, alignment)

    starts, keys = split.query_start.tolist(), split.key_lengths.tolist()
    sizes = [end - start for start, end in zip(starts, starts[1:])]
    # (first position, rows) of each new request: its rows are the last ones of its keys.
    assert [(key - size, size) for key, size in zip(keys, sizes)] == pieces
    assert starts[0] == 0 and starts[-1] == query_start[-1]
    owners = split.owner.tolist()
    assert owners == sorted(owners) and set(owners) == set(range(len(requests)))


def test_fa3_request_calls_separate_one_row_runs_and_bound_each_call():
    # Request sizes: three one-row requests, two longer ones, four one-row ones.
    query_start = [0, 1, 2, 3, 7, 10, 11, 12, 13, 14]

    calls = fa3_request_calls(query_start, one_row_calls=True, max_requests=3)

    assert [tuple(call) for call in calls] == [(0, 3, True), (3, 5, False), (5, 8, True), (8, 9, True)]
    assert [tuple(call) for call in fa3_request_calls(query_start, one_row_calls=False, max_requests=5)] == [
        (0, 5, False),
        (5, 9, False),
    ]
