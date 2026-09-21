import numpy as np

from speechofficesim.pipelines.vad import compute_vad_profile


def test_compute_vad_profile_detects_loud_then_silent_segments():
    sr = 16000
    window_ms = 50
    win_len = int(round(sr * window_ms / 1000.0))

    n_loud_frames = 5
    n_silent_frames = 5
    t = np.arange(win_len * n_loud_frames) / sr
    loud = np.sin(2 * np.pi * 440 * t)
    silent = np.zeros(win_len * n_silent_frames)
    audio = np.concatenate([loud, silent])

    profile = compute_vad_profile(audio, sr, window_ms=window_ms)

    assert len(profile) == n_loud_frames + n_silent_frames
    assert np.all(profile[:n_loud_frames] == 1)
    assert np.all(profile[n_loud_frames:] == 0)


def test_compute_vad_profile_handles_silence_without_dividing_by_zero():
    sr = 8000
    audio = np.zeros(sr)
    profile = compute_vad_profile(audio, sr)
    assert np.all(profile == 0)
