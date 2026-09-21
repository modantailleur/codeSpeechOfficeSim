import random
from collections import namedtuple

import numpy as np

from speechofficesim.pipelines.multispeaker_lengths import (
    clean_callhome_tokens,
    clean_text,
    match_vctk_to_callhome,
)

Token = namedtuple("Token", ["word"])


def test_clean_callhome_tokens_filters_noise_and_lowercases():
    tokens = [
        Token("Hello"),
        Token("&=laughs"),
        Token("foo@bar"),
        Token("xxx"),
        Token("+ing"),
        Token("..."),
        Token("WORLD!"),
        Token(None),
    ]
    assert clean_callhome_tokens(tokens) == ["hello", "world"]


def test_clean_text_filters_punctuation_and_lowercases():
    text = "Hello, world! ... it's 2026 -- great."
    assert clean_text(text) == ["hello", "world", "it's", "great"]


def test_match_vctk_to_callhome_covers_every_file_exactly_once():
    random.seed(0)
    np.random.seed(0)

    vctk_utts = []
    for spk in ("p1", "p2", "p3"):
        for i in range(10):
            length = (i % 5) + 1
            vctk_utts.append((["w"] * length, length, f"{spk}_{i:03d}.txt"))

    target_lengths = np.array([2, 4, 6, 8, 10, 3, 5, 7])

    pseudo_lengths, groups = match_vctk_to_callhome(vctk_utts, target_lengths, n=len(target_lengths))

    all_fnames_in = [fname for _, _, fname in vctk_utts]
    all_fnames_out = [fname for group in groups for fname in group]

    assert sorted(all_fnames_out) == sorted(all_fnames_in)
    assert len(pseudo_lengths) == len(groups)
