"""Small audio/text I/O helpers shared across the reformatting pipelines."""
from pathlib import Path

import numpy as np
import soundfile as sf

from .acoustics import compute_LAeq


def load_mono_audio(path):
    """Read an audio file and downmix to mono if needed."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def audio_duration(audio, sr):
    return len(audio) / sr


def make_silence(sr, duration):
    return np.zeros(int(duration * sr))


def normalize_laeq(audio, sr, target_laeq, source_name=""):
    """Scale audio so its LAeq matches target_laeq, then rescale if that would clip."""
    audio_dba = compute_LAeq(audio, sr)
    scale = 10 ** ((target_laeq - audio_dba) / 20)
    audio = audio * scale

    max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
        print(f"   ⚠️ Audio normalized to avoid clipping for {source_name} (max abs was {max_abs:.6f})")

    return audio


def pad_to_length_centered(audio, sr, target_len):
    target_samples = int(target_len * sr)
    pad_total = target_samples - len(audio)
    if pad_total <= 0:
        return audio
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.concatenate([np.zeros(pad_left), audio, np.zeros(pad_right)])


def apply_fade(audio, sr, fade_dur=0.01):
    n_fade = min(int(sr * fade_dur), len(audio) // 2)
    if n_fade <= 0:
        return audio
    audio = audio.copy()
    fade_curve = np.linspace(0.0, 1.0, n_fade)
    audio[:n_fade] *= fade_curve
    audio[-n_fade:] *= fade_curve[::-1]
    return audio


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_txt(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
