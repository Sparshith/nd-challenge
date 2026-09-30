from pathlib import Path
from types import SimpleNamespace
import json

import chess
import numpy as np
import pytest

from starter.check import Entry, play


@pytest.mark.parametrize("token", ["e5", "<bos>", "<eos>", "<unk>", "--", "e2e4", "Ng1f3", "bad"])
def test_invalid_top_token_cannot_fall_back_to_a_legal_move(token):
    entry = object.__new__(Entry)
    entry.vocabulary = [token, "e4"]
    entry.tokens = lambda sans: []
    entry.logits = lambda tokens: np.array([10.0, 9.0])
    assert entry.move(chess.Board(), []) is None


def test_lowest_token_id_breaks_ties():
    entry = object.__new__(Entry)
    entry.vocabulary = ["d4", "e4"]
    entry.tokens = lambda sans: []
    entry.logits = lambda tokens: np.array([1.0, 1.0])
    assert entry.move(chess.Board(), []) == chess.Move.from_uci("d2d4")


def test_mate_suffix_is_optional():
    entry = object.__new__(Entry)
    entry.vocabulary = ["Qh4"]
    entry.tokens = lambda sans: []
    entry.logits = lambda tokens: np.array([1.0])
    board = chess.Board()
    for san in "f3 e5 g4".split():
        board.push_san(san)
    assert entry.move(board, []) == board.parse_san("Qh4#")


def test_match_records_invalid_predictions_as_losses(capsys):
    invalid = SimpleNamespace(name="invalid", context=512, move=lambda board, sans: None)
    legal = SimpleNamespace(name="legal", context=512, move=lambda board, sans: next(iter(board.legal_moves)))
    play(invalid, legal, 2)
    assert "invalid vs legal: 0 - 2" in capsys.readouterr().out


def test_browser_openings_are_legal_and_supported_by_the_starter():
    root = Path(__file__).resolve().parents[1]
    vocabulary = set(json.loads((root / 'starter/vocabulary.json').read_text()))
    rows = [line for line in (root / 'dashboard/openings.tsv').read_text().splitlines() if line and not line.startswith('#')][1:]
    assert len(rows) == 200
    for row in rows:
        board = chess.Board()
        for san in row.split('\t')[2].split():
            assert san in vocabulary
            board.push_san(san)
