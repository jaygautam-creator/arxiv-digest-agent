"""CLI error handling.

Author: Jay Gautam (jaygautam561@gmail.com)
Project: 8byte Assessment
"""

import sys
from unittest.mock import patch

import pytest

from arxiv_digest import cli


def test_corrupt_session_file_exits_cleanly(tmp_path, capsys):
    bad = tmp_path / "session.json"
    bad.write_text('{"broken": ')
    with (
        patch.object(sys, "argv", ["arxiv_digest", "--session", str(bad), "--mock"]),
        pytest.raises(SystemExit) as exit_,
    ):
        cli.main()
    assert exit_.value.code == 1
    assert "Could not load session file" in capsys.readouterr().out


def test_ctrl_c_exits_cleanly(capsys):
    with patch.object(cli, "run", side_effect=KeyboardInterrupt), pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 130
    assert "Interrupted" in capsys.readouterr().out
