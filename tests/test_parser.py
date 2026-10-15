import pytest
import datetime
from scripts.utils import parse_single_file, parse_all_files

def test_parse_private_chat():
    records = parse_single_file("tests/data/private_chat.html")
    assert len(records) == 11

    # Check first message
    assert records[0]["sender"] == "Alice Example"
    assert records[0]["text"] == "Hi Bob!"
    assert records[0]["timestamp"] == datetime.datetime(2023, 1, 1, 10, 0, 0)

    # Check consecutive messages
    assert records[1]["sender"] == "Alice Example"
    assert records[1]["text"] == "How are you doing today?"
    assert records[1]["timestamp"] == datetime.datetime(2023, 1, 1, 10, 1, 0)

    assert records[2]["sender"] == "Alice Example"
    assert records[2]["text"] == "Hope you are well!"
    assert records[2]["timestamp"] == datetime.datetime(2023, 1, 1, 10, 2, 0)

    # Check new sender
    assert records[3]["sender"] == "Bob Example"
    assert records[3]["text"] == "Hi Alice, I am good!"
    assert records[3]["timestamp"] == datetime.datetime(2023, 1, 1, 10, 5, 0)


def test_parse_group_chat():
    records = parse_single_file("tests/data/group_chat.html")
    assert len(records) == 5

    senders = set(r["sender"] for r in records)
    assert senders == {"Alice Example", "Bob Example", "Charlie Fake", "Dave Dummy"}


def test_parse_all_files_overlap():
    df = parse_all_files("tests/data/overlap*.html")

    # overlap1 has Message 1 and 2, overlap2 has Message 2 and 3.
    # So total unique should be 3 messages.
    assert len(df) == 3
    assert list(df["text"]) == ["Message 1", "Message 2", "Message 3"]


@pytest.mark.xfail(reason="Forwarded messages misattribute sender to 'Forwarded from X' instead of current_sender", strict=True)
def test_parse_forwarded_message():
    records = parse_single_file("tests/data/edge_cases.html")
    # First parsed record should be the forwarded message, because service messages are ignored.
    # Current behavior sets sender to "Forwarded from Jane" instead of None or the actual sender.
    assert records[0]["sender"] != "Forwarded from Jane"

def test_parse_edge_cases():
    records = parse_single_file("tests/data/edge_cases.html")

    texts = [r["text"] for r in records]
    # Service message is correctly ignored.
    assert "Alice Example pinned a message" not in texts

    # Mixed text and link is parsed (current implementation uses separator=" " so we have a space before "!")
    assert "Check this: Link !" in texts

    # Non-ASCII is parsed.
    assert "Hello 🌍! привет" in texts


