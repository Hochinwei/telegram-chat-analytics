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


def test_parse_forwarded_message():
    records = parse_single_file("tests/data/forwarded_cases.html")
    # Check messages
    assert len(records) == 4

    # 1. Normal message before forwarded message
    assert records[0]["sender"] == "Alice Example"
    assert records[0]["is_forwarded"] is False

    # 2. Forwarded message with its own outer from_name
    assert records[1]["sender"] == "Alice Example"
    assert records[1]["text"] == "This is forwarded text!"
    assert records[1]["is_forwarded"] is True

    # 3. Forwarded continuation message with no outer from_name (should keep previous sender)
    assert records[2]["sender"] == "Alice Example"
    assert records[2]["text"] == "This is the second forwarded text!"
    assert records[2]["is_forwarded"] is True

    # 4. Normal message immediately after a forwarded message (verifying current_sender wasn't contaminated by original author)
    assert records[3]["sender"] == "Alice Example"
    assert records[3]["text"] == "And this is a normal message."
    assert records[3]["is_forwarded"] is False

    # Additionally verify edge_cases.html that original author does not appear in sender
    records_edge = parse_single_file("tests/data/edge_cases.html")
    for r in records_edge:
        assert not r["sender"].startswith("Forwarded from")
        assert r["sender"] != "Jane Fake"

def test_parse_edge_cases():
    records = parse_single_file("tests/data/edge_cases.html")

    texts = [r["text"] for r in records]
    # Service message is correctly ignored.
    assert "Alice Example pinned a message" not in texts

    # Mixed text and link is parsed (current implementation uses separator=" " so we have a space before "!")
    assert "Check this: Link !" in texts

    # Non-ASCII is parsed.
    assert "Hello 🌍! привет" in texts


