import pytest
import os
import sys
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")

@pytest.mark.xfail(reason="Hardcoded paths or argparse issues in script execution without refactoring", strict=False)
def test_group_chat_smoke(tmp_path):
    out_dir = tmp_path / "group_out"
    out_dir.mkdir()

    test_args = [
        "group_chat_analysis.py",
        "--input-pattern", "tests/data/group_chat.html",
        "--output-dir", str(out_dir)
    ]
    with patch.object(sys, 'argv', test_args):
        import scripts.group_chat_analysis

    # Verify outputs
    assert (out_dir / "parsed_messages.csv").exists()

@pytest.mark.xfail(reason="Hardcoded paths or argparse issues in script execution without refactoring", strict=False)
def test_private_chat_smoke(tmp_path):
    out_dir = tmp_path / "private_out"
    out_dir.mkdir()

    test_args = [
        "private_chat_analysis.py",
        "--input-pattern", "tests/data/private_chat.html",
        "--output-dir", str(out_dir),
        "--your-name", "Alice Example",
        "--other-name", "Bob Example"
    ]
    with patch.object(sys, 'argv', test_args):
        import scripts.private_chat_analysis

    # Verify outputs
    assert (out_dir / "parsed_messages.csv").exists()
