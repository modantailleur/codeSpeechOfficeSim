"""Diagnostic: report the combined LAeq of the keyboard + office ambience source clips.

Ported from the original compute_office_sound_level.py script.
"""
import librosa
import numpy as np
import soundfile as sf

from ..acoustics import compute_LAeq


def run(
    keyboard_path="./audio/399823__bonnyorbit__keyboard-typing-in-office.wav",
    office_path="./audio/541117__chelly01__office-ambience.wav",
    sr=48000,
    output_path="combined_audio.wav",
):
    y_key, _ = librosa.load(keyboard_path, sr=sr, mono=False)
    y_off, _ = librosa.load(office_path, sr=sr, mono=False)
    n_repeat = int(np.ceil(y_off.shape[1] / y_key.shape[1]))
    y_key_repeated = np.tile(y_key, (1, n_repeat))[:, :y_off.shape[1]]
    y = (y_key_repeated + y_off) / 2.0
    laeq = compute_LAeq(y, sr)
    print(f"Combined audio: LAeq ≈ {laeq:.2f} dBA")

    sf.write(output_path, y.T, sr)
