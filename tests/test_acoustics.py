import numpy as np
import pytest

from speechofficesim.acoustics import RoomAcousticProcessor, compute_LAeq


def _sine(freq, sr, duration=1.0, amplitude=1.0):
    t = np.arange(int(sr * duration)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_a_weighting_attenuates_low_frequencies_more_than_1khz():
    # This filter's gain is on an internal/arbitrary scale (not calibrated to
    # real-world dBA — see the very negative LAeq targets used throughout the
    # pipeline), but it should still roll off bass more than a 1kHz tone.
    sr = 48000
    low_tone = _sine(50, sr, duration=1.0, amplitude=0.5)
    mid_tone = _sine(1000, sr, duration=1.0, amplitude=0.5)
    assert compute_LAeq(low_tone, sr) < compute_LAeq(mid_tone, sr)


def test_compute_laeq_silence_is_very_low():
    sr = 16000
    silence = np.zeros(sr)
    assert compute_LAeq(silence, sr) < -200


def test_compute_laeq_averages_multichannel_signals():
    sr = 16000
    tone = _sine(1000, sr, duration=0.5, amplitude=0.3)
    mono_laeq = compute_LAeq(tone, sr)
    stereo = np.stack([tone, tone])  # channel-first, like pyroomacoustics output
    assert compute_LAeq(stereo, sr) == pytest.approx(mono_laeq, abs=1e-9)


def test_room_acoustic_processor_produces_stereo_output():
    sr = 8000
    signal = _sine(440, sr, duration=0.05, amplitude=0.8)
    processor = RoomAcousticProcessor(sr, room_dim=[5, 5, 3])
    out = processor.process(signal, pan=0.0)
    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert np.all(np.isfinite(out))
