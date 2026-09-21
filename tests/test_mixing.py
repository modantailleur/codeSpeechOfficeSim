import numpy as np
import pytest

from speechofficesim.acoustics import compute_LAeq
from speechofficesim.pipelines.mixing import EBRS, SARS, mix_scene


def _noise(rng, n_samples, amplitude):
    return amplitude * rng.standard_normal(n_samples)


@pytest.mark.parametrize("ebr", list(EBRS.keys()))
@pytest.mark.parametrize("sar", list(SARS.keys()))
def test_mix_scene_shape_and_finiteness(ebr, sar):
    sr = 16000
    rng = np.random.default_rng(0)
    speech = _noise(rng, sr, 0.2)
    events = _noise(rng, sr, 0.05)
    bg = _noise(rng, sr, 0.01)

    mixed = mix_scene(speech, events, bg, sr, ebr=ebr, sar=sar)

    assert mixed.shape == speech.shape
    assert np.all(np.isfinite(mixed))


@pytest.mark.parametrize("sar", list(SARS.keys()))
def test_mix_scene_respects_speech_to_ambient_ratio(sar):
    """The ambient bed recovered from the mix (y_mixed - y_speech_scaled) should
    sit `sars[sar]` dB below the speech, per the documented SAR contract."""
    sr = 16000
    rng = np.random.default_rng(1)
    speech = _noise(rng, sr, 0.2)
    events = _noise(rng, sr, 0.05)
    bg = _noise(rng, sr, 0.01)

    mixed = mix_scene(speech, events, bg, sr, ebr="mid", sar=sar)
    recovered_ambient = mixed - speech

    laeq_speech = compute_LAeq(speech, sr)
    laeq_ambient = compute_LAeq(recovered_ambient, sr)

    assert (laeq_speech - laeq_ambient) == pytest.approx(SARS[sar], abs=0.05)
