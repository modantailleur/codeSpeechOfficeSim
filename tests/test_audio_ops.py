import numpy as np
import pytest

from speechofficesim.acoustics import compute_LAeq
from speechofficesim.audio_ops import (
    apply_fade,
    make_silence,
    normalize_laeq,
    pad_to_length_centered,
    load_txt,
    save_txt,
)


def test_make_silence_length_and_values():
    sr = 16000
    silence = make_silence(sr, 0.25)
    assert len(silence) == int(0.25 * sr)
    assert np.all(silence == 0)


def test_pad_to_length_centered_pads_symmetrically():
    sr = 100
    audio = np.ones(50)
    padded = pad_to_length_centered(audio, sr, target_len=1.0)  # 100 samples
    assert len(padded) == 100
    pad_left = 25
    assert np.all(padded[:pad_left] == 0)
    assert np.all(padded[pad_left:pad_left + 50] == 1)
    assert np.all(padded[pad_left + 50:] == 0)


def test_pad_to_length_centered_noop_when_already_long_enough():
    audio = np.ones(200)
    padded = pad_to_length_centered(audio, sr=100, target_len=1.0)  # target 100 samples
    assert padded is audio


def test_normalize_laeq_hits_target_when_no_clipping_needed():
    sr = 16000
    t = np.arange(sr) / sr
    audio = 0.9 * np.sin(2 * np.pi * 1000 * t)
    # Pick a target quieter than the source's own level so no anti-clip
    # rescaling kicks in (this filter's LAeq isn't real-world-calibrated dBA).
    target = compute_LAeq(audio, sr) - 10.0
    out = normalize_laeq(audio, sr, target)
    assert compute_LAeq(out, sr) == pytest.approx(target, abs=0.1)
    assert np.max(np.abs(out)) <= 1.0


def test_normalize_laeq_anti_clips_when_target_too_loud():
    sr = 16000
    t = np.arange(sr) / sr
    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)
    out = normalize_laeq(audio, sr, target_laeq=20.0)  # unreasonably loud target
    assert np.max(np.abs(out)) == pytest.approx(1.0)


def test_apply_fade_ramps_edges_and_preserves_middle():
    sr = 1000
    audio = np.ones(200)
    faded = apply_fade(audio, sr, fade_dur=0.05)  # 50 samples
    assert faded[0] == pytest.approx(0.0)
    assert faded[-1] == pytest.approx(0.0)
    assert faded[100] == pytest.approx(1.0)
    assert np.all(np.diff(faded[:50]) >= 0)


def test_apply_fade_noop_on_too_short_audio():
    audio = np.array([1.0])
    assert np.array_equal(apply_fade(audio, sr=1000, fade_dur=0.01), audio)


def test_load_save_txt_round_trip_and_creates_parent_dirs(tmp_path):
    nested_path = tmp_path / "a" / "b" / "c.txt"
    save_txt(nested_path, "  hello world  ")
    assert nested_path.exists()
    assert load_txt(nested_path) == "hello world"
