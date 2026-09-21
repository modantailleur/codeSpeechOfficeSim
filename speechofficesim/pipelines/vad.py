"""Compute ground-truth speech-activity annotations from the ebr-none-sar-none split.

Ported from the original compute_groundtruth_vad.py script.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOW_MS = 50
THRESHOLD_DB = -36.0  # activity threshold relative to peak after normalization
EPS = 1e-12


def compute_vad_profile(audio, sr, window_ms=WINDOW_MS, threshold_db=THRESHOLD_DB):
    """Peak-normalize, frame, and threshold an audio signal into a binary activity profile."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak  # peak normalize

    win_len = int(round(sr * window_ms / 1000.0))
    if win_len <= 0:
        raise ValueError("Invalid window length for the given sample rate")

    n_frames = int(np.ceil(len(audio) / win_len))
    pad_len = n_frames * win_len - len(audio)
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len), mode="constant")

    frames = audio.reshape(n_frames, win_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_db = 20.0 * np.log10(rms + EPS)

    # 1 = active signal, 0 = inactive
    return (frame_db >= threshold_db).astype(np.uint8)


def run(dataset_path="../SOS-1SP"):
    dataset_path = Path(dataset_path)
    root = dataset_path / "ebr-none-sar-none"
    out_dir = dataset_path / "vadgt"
    out_dir.mkdir(parents=True, exist_ok=True)

    flac_files = sorted(root.rglob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files")

    for path in flac_files:
        try:
            audio, sr = sf.read(path, always_2d=True)
            audio = audio.mean(axis=1)  # mono

            signal_profile = compute_vad_profile(audio, sr)

            parts = path.stem.split("__")
            name = parts[1] if len(parts) > 1 else path.stem
            out_path = out_dir / f"{name}.npy"
            np.save(out_path, signal_profile)

            print(f"Saved {out_path} | frames={len(signal_profile)}")

        except Exception as e:
            print(f"ERROR processing {path}: {e}")
