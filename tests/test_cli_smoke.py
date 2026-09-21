import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SUBCOMMANDS = [
    "monospeaker-reformat",
    "multispeaker-lengths",
    "multispeaker-group",
    "multispeaker-conversations",
    "mix",
    "vad",
    "office-level",
]


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "speechofficesim", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_top_level_help_lists_all_subcommands():
    result = _run("--help")
    assert result.returncode == 0
    for cmd in SUBCOMMANDS:
        assert cmd in result.stdout


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help_exits_cleanly(cmd):
    result = _run(cmd, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_missing_subcommand_fails_with_usage_error():
    result = _run()
    assert result.returncode != 0
