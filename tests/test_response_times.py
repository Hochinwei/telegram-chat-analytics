import datetime

# To test plotting smoke test
import matplotlib
import pandas as pd
import pytest

from scripts.private_chat_analysis import plot_response_times
from scripts.utils import compute_response_times

matplotlib.use("Agg")
import os


# Reference implementation of the old loop for equivalence testing
def _old_loop_response_times(df):
    if df.empty:
        return pd.DataFrame()
    df_s = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    rows = []
    for i in range(1, len(df_s)):
        curr = df_s.loc[i]
        prev = df_s.loc[i - 1]
        if curr["sender"] != prev["sender"]:
            gap = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 60
            if 0 < gap < 1440:
                rows.append({"responder": curr["sender"],
                             "response_min": gap,
                             "month": curr["month"] if "month" in curr else None})
    rdf = pd.DataFrame(rows)
    if "month" not in df.columns and not rdf.empty and "month" in rdf.columns:
        rdf = rdf.drop(columns=["month"])
    return rdf

def create_mock_df(data):
    """Helper to create test DataFrame"""
    df = pd.DataFrame(data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # add a month period just like the real app
        df["month"] = df["timestamp"].dt.to_period("M")
    else:
        df = pd.DataFrame(columns=["sender", "timestamp", "month"])
    return df

@pytest.fixture
def chat_df():
    # 1. Alice 10:00, 10:01, 10:02 then Bob 10:05. Response 3 min for Bob.
    # 2. Two-way alternation produces the expected sequence.
    # 3. Gap exactly equal to the cap is excluded; one minute under is included.
    # 4. Same-timestamp reply (gap 0) is included.
    data = [
        {"sender": "Alice", "timestamp": "2023-01-01 10:00:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:01:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:02:00"},
        {"sender": "Bob",   "timestamp": "2023-01-01 10:05:00"}, # Bob response 3 mins
        {"sender": "Alice", "timestamp": "2023-01-01 10:06:00"}, # Alice response 1 min
        {"sender": "Bob",   "timestamp": "2023-01-01 10:06:00"}, # Bob response 0 mins
        {"sender": "Alice", "timestamp": "2023-01-02 10:05:00"}, # Alice response 1439 mins
        {"sender": "Bob",   "timestamp": "2023-01-03 10:05:00"}, # Bob response 1440 mins (excluded default)
        {"sender": "Alice", "timestamp": "2023-01-06 10:05:00"}, # Alice response 3 days (excluded default)
    ]
    return create_mock_df(data)


def test_basic_response_time(chat_df):
    # Test 1 & 2: Alice 10:00-10:02, Bob 10:05
    rdf, excluded = compute_response_times(chat_df, return_excluded=True)

    # We expect:
    # row 0: Bob, 3.0 min (from 10:02 to 10:05)
    # row 1: Alice, 1.0 min (from 10:05 to 10:06)
    # row 2: Bob, 0.0 min (from 10:06 to 10:06)
    # row 3: Alice, 1439.0 min (from 10:06 to Jan 2 10:05)

    assert len(rdf) == 4
    assert excluded == 2 # The 1440 min and the 3-day multi-day gaps

    assert rdf.iloc[0]["responder"] == "Bob"
    assert rdf.iloc[0]["response_min"] == 3.0

    assert rdf.iloc[1]["responder"] == "Alice"
    assert rdf.iloc[1]["response_min"] == 1.0

    assert rdf.iloc[2]["responder"] == "Bob"
    assert rdf.iloc[2]["response_min"] == 0.0

    assert rdf.iloc[3]["responder"] == "Alice"
    assert rdf.iloc[3]["response_min"] == 1439.0

def test_custom_max_gap(chat_df):
    # Test 10: A custom max_gap_minutes changes the outcome
    rdf, excluded = compute_response_times(chat_df, max_gap_minutes=2000, return_excluded=True)
    # The 1440 min should now be included, but 3 days (4320 mins) still excluded
    assert len(rdf) == 5
    assert excluded == 1

    assert rdf.iloc[4]["responder"] == "Bob"
    assert rdf.iloc[4]["response_min"] == 1440.0

def test_single_sender():
    # Test 6: Single-sender chat returns an empty result with the right columns.
    df = create_mock_df([
        {"sender": "Alice", "timestamp": "2023-01-01 10:00:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:01:00"}
    ])
    rdf, excluded = compute_response_times(df, return_excluded=True)
    assert rdf.empty
    assert list(rdf.columns) == ["responder", "response_min", "month"]
    assert excluded == 0

def test_empty_and_one_row():
    # Test 7: Empty and one-row DataFrames do not raise.
    df_empty = create_mock_df([])
    rdf1, excluded1 = compute_response_times(df_empty, return_excluded=True)
    assert rdf1.empty
    assert excluded1 == 0
    assert list(rdf1.columns) == ["responder", "response_min", "month"]

    df_one = create_mock_df([{"sender": "Alice", "timestamp": "2023-01-01 10:00:00"}])
    rdf2, excluded2 = compute_response_times(df_one, return_excluded=True)
    assert rdf2.empty
    assert excluded2 == 0
    assert list(rdf2.columns) == ["responder", "response_min", "month"]

def test_input_not_mutated(chat_df):
    # Test 8: Input DataFrame is not mutated.
    orig_shape = chat_df.shape
    orig_values = chat_df.values.copy()

    compute_response_times(chat_df)

    assert chat_df.shape == orig_shape
    pd.testing.assert_frame_equal(chat_df, pd.DataFrame(orig_values, columns=chat_df.columns))

def test_unsorted_input():
    # Test 9: Unsorted input gives the same result as sorted input.
    # Note: Using distinct timestamps so stable sort behaves the same independent of initial order.
    # When timestamps are identical, stable sort preserves original order, so shuffled
    # data can produce different alternating sequences.
    data = [
        {"sender": "Alice", "timestamp": "2023-01-01 10:00:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:01:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:02:00"},
        {"sender": "Bob",   "timestamp": "2023-01-01 10:05:00"},
        {"sender": "Alice", "timestamp": "2023-01-01 10:06:00"},
        {"sender": "Bob",   "timestamp": "2023-01-01 10:07:00"},
        {"sender": "Alice", "timestamp": "2023-01-02 10:05:00"},
        {"sender": "Bob",   "timestamp": "2023-01-03 10:05:00"},
        {"sender": "Alice", "timestamp": "2023-01-06 10:05:00"},
    ]
    df = create_mock_df(data)

    sorted_rdf = compute_response_times(df)

    # Shuffle the input randomly
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    unsorted_rdf = compute_response_times(shuffled_df)

    # Ensure they have identical rows when sorted similarly
    sorted_rdf = sorted_rdf.sort_values(["responder", "response_min"]).reset_index(drop=True)
    unsorted_rdf = unsorted_rdf.sort_values(["responder", "response_min"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(sorted_rdf, unsorted_rdf)

def test_equivalence_with_old_loop():
    # Test 11: Equivalence with the old loop (apart from 0-gap).
    import numpy as np
    np.random.seed(42)

    # Create random synthetic DataFrame avoiding exact-zero gaps
    n = 200
    timestamps = [datetime.datetime(2023, 1, 1, 0, 0, 0)]
    for _ in range(n-1):
        # ensure gap is always > 0 by adding at least 1 second
        gap_sec = np.random.randint(1, 100000)
        timestamps.append(timestamps[-1] + datetime.timedelta(seconds=int(gap_sec)))

    senders = np.random.choice(["Alice", "Bob", "Charlie"], n)
    df = create_mock_df([{"timestamp": t, "sender": s} for t, s in zip(timestamps, senders)])

    # Run new vectorized func
    new_rdf = compute_response_times(df)

    # Run old loop
    old_rdf = _old_loop_response_times(df)

    # They should be identical since there are no 0 gaps and we used the same logic
    if old_rdf.empty:
        assert new_rdf.empty
    else:
        # Sort and compare
        new_rdf = new_rdf.sort_values(by=["responder", "response_min"]).reset_index(drop=True)
        old_rdf = old_rdf.sort_values(by=["responder", "response_min"]).reset_index(drop=True)

        # We need to coerce month column to string to compare properly as old might have object, new Period
        new_rdf["month"] = new_rdf["month"].astype(str)
        old_rdf["month"] = old_rdf["month"].astype(str)

        pd.testing.assert_frame_equal(new_rdf, old_rdf)

def test_smoke_plot_response_times(tmp_path, chat_df):
    # Test 12: A smoke test that plot_response_times writes R1 and R2 PNGs
    colors = {"Alice": "#ff0000", "Bob": "#0000ff"}

    # Call the plotting function
    plot_response_times(chat_df, colors, str(tmp_path), "Alice", "Bob")

    # Check that the files were created
    assert os.path.exists(tmp_path / "R1_response_time_dist.png")
    assert os.path.exists(tmp_path / "R2_response_time_trend.png")
