"""Audit a *generated* SOS-1SP/SOS-2SP dataset for EBR/SAR coherence, without needing VCTK.

Idea
----
For a given (speaker, segment, pan) triple, mixing.py renders the SAME
leveled, room-processed speech signal `y_speech_scaled` across all 10 noise
conditions -- only the `ebr-none-sar-none` file skips adding any ambient
content, so it *is* `y_speech_scaled` exactly. That gives us, for free, a
per-file reference we can use to check the other 9 conditions without ever
touching VCTK:

  y_mixed = y_speech_scaled + y_amb_scaled
  =>  y_amb_scaled (recovered) = y_mixed - y_speech_scaled

1. SAR check (exact): mix_scene sets
       target_amb_laeq = laeq_speech - SARS[sar]
   so LAeq(y_speech_scaled) - LAeq(y_amb_scaled_recovered) should equal
   SARS[sar] to within numerical precision. This needs nothing but the
   dataset's own files.

2. EBR check (best-effort): y_amb_scaled_recovered = scale_amb * (scale_events
   * y_events_seg + y_bg_seg), i.e. a linear combination of the *raw* events
   and background source clips (still bundled in ./audio/, so no VCTK needed
   either). We locate the most likely time offset of each source clip inside
   the recovered ambient bed via FFT cross-correlation, then solve a 2
   unknown least-squares fit for the two implied scale factors. Because both
   terms share the same `scale_amb` factor, it cancels in their LAeq
   difference, so the recovered events-vs-background gap should equal
   EBRS[ebr]. This is inherently less reliable than the SAR check (stationary
   background noise doesn't localize as cleanly as the events clip's
   transients, and "low" EBR events are ~50dB below background, i.e. close to
   undetectable by design), so a confidence score is reported alongside it,
   and a condition whose events term explains no meaningful unique variance
   is reported as inconclusive rather than a false failure.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from ..acoustics import compute_LAeq
from .mixing import EBRS, SARS, TARGET_SPEECH_LAEQ

SAR_TOLERANCE_DB = 0.5
EBR_TOLERANCE_DB = 1.0
CONDITIONS = [(ebr, sar) for ebr in EBRS for sar in SARS]


def load_channel_first(path):
    """soundfile reads (samples, channels); the pipeline works in (channels, samples)."""
    audio, sr = sf.read(path, always_2d=True)
    return audio.T, sr


def load_source_mono(path, sr):
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=False)
    return audio, audio.mean(axis=0)


def first_n_ids(dataset_root, n):
    clean_root = dataset_root / "ebr-none-sar-none" / "pan_0"
    ids = sorted(p.name for p in clean_root.iterdir() if p.is_dir())
    return ids[:n]


def first_m_clean_files(dataset_root, speaker_id, m):
    speaker_dir = dataset_root / "ebr-none-sar-none" / "pan_0" / speaker_id
    files = sorted(speaker_dir.glob("*.flac"))
    return files[:m]


def condition_path(clean_path, ebr, sar):
    return Path(str(clean_path).replace("ebr-none-sar-none", f"ebr-{ebr}-sar-{sar}"))


def localize_offset(candidate_mono, source_mono):
    """Best time offset (in samples) of candidate_mono inside source_mono, via
    normalized FFT cross-correlation, plus a confidence score (peak height
    relative to the correlation function's typical spread)."""
    if len(source_mono) < len(candidate_mono):
        return None, 0.0

    candidate_energy = np.sqrt(np.sum(candidate_mono**2))
    if candidate_energy == 0:
        return None, 0.0

    corr = correlate(source_mono, candidate_mono, mode="valid", method="fft")

    # Normalize by the local energy of the source window at each lag, so the
    # score is comparable across offsets and files (a Pearson-correlation-like
    # quantity, in [-1, 1] ideally, though our windows aren't unit-energy).
    # A sliding window sum-of-squares via cumsum, not np.convolve, which is an
    # O(N*M) direct convolution -- too slow for multi-million-sample inputs.
    cumsum_sq = np.cumsum(np.concatenate([[0.0], source_mono**2]))
    n = len(candidate_mono)
    window_energy_sq = cumsum_sq[n:] - cumsum_sq[:-n]
    window_energy = np.sqrt(window_energy_sq)
    window_energy[window_energy == 0] = np.inf
    normalized = corr / (window_energy * candidate_energy)

    best_offset = int(np.argmax(np.abs(normalized)))
    peak = float(np.abs(normalized[best_offset]))
    background = float(np.median(np.abs(normalized)))
    confidence = peak / background if background > 0 else 0.0
    return best_offset, confidence


def check_sar(clean_audio, noisy_audio, sr, sar_key):
    laeq_speech = compute_LAeq(clean_audio, sr)
    y_amb_recovered = noisy_audio - clean_audio
    laeq_amb = compute_LAeq(y_amb_recovered, sr)
    measured_sar = laeq_speech - laeq_amb
    expected_sar = SARS[sar_key]
    diff = measured_sar - expected_sar
    return {
        "laeq_speech": laeq_speech,
        "measured_sar": measured_sar,
        "expected_sar": expected_sar,
        "diff": diff,
        "pass": abs(diff) <= SAR_TOLERANCE_DB,
    }


def evaluate_ebr(y_amb_recovered, sr, events_full, events_mono, bg_full, bg_mono, ebr_key):
    seg_len = y_amb_recovered.shape[1]
    amb_mono = y_amb_recovered.mean(axis=0)

    off_events, conf_events = localize_offset(amb_mono, events_mono)
    off_bg, conf_bg = localize_offset(amb_mono, bg_mono)

    if off_events is None or off_bg is None:
        return {"pass": None, "reason": "segment longer than a source clip"}

    events_seg = events_full[:, off_events:off_events + seg_len]
    bg_seg = bg_full[:, off_bg:off_bg + seg_len]

    # y_amb_recovered ~= A * events_seg + B * bg_seg, solved jointly over both
    # stereo channels (A, B are scalars applied identically to both channels
    # in the original pipeline).
    X = np.stack([events_seg.reshape(-1), bg_seg.reshape(-1)], axis=1)
    y = y_amb_recovered.reshape(-1)
    (A, B), *_ = np.linalg.lstsq(X, y, rcond=None)

    residual = y - X @ np.array([A, B])
    y_energy = max(np.sum(y**2), 1e-30)
    fit_r2 = 1.0 - np.sum(residual**2) / y_energy

    # Unique variance the events regressor explains on top of background
    # alone: when EBR is very negative (events deliberately near-inaudible,
    # e.g. "low" = -50dB), the events term contributes essentially nothing
    # to the mix, so the fitted A is just noise-fitting -- a plausible-looking
    # but meaningless number, even though the *overall* fit (dominated by
    # background) still looks good. Catch that case explicitly.
    (B_only,), *_ = np.linalg.lstsq(bg_seg.reshape(-1, 1), y, rcond=None)
    residual_bg_only = y - bg_seg.reshape(-1) * B_only
    r2_bg_only = 1.0 - np.sum(residual_bg_only**2) / y_energy
    incremental_r2 = fit_r2 - r2_bg_only

    laeq_events = compute_LAeq(A * events_seg, sr)
    laeq_bg = compute_LAeq(B * bg_seg, sr)
    measured_ebr = laeq_events - laeq_bg
    expected_ebr = EBRS[ebr_key]
    diff = measured_ebr - expected_ebr

    # Low confidence in the source-clip localization, a poor linear fit, or a
    # negligible unique contribution from the events term, all mean the EBR
    # estimate itself isn't trustworthy -- report but don't fail on it.
    reliable = conf_events > 3.0 and conf_bg > 3.0 and fit_r2 > 0.9 and incremental_r2 > 1e-4

    return {
        "measured_ebr": measured_ebr,
        "expected_ebr": expected_ebr,
        "diff": diff,
        "conf_events": conf_events,
        "conf_bg": conf_bg,
        "fit_r2": fit_r2,
        "incremental_r2": incremental_r2,
        "reliable": reliable,
        "pass": (abs(diff) <= EBR_TOLERANCE_DB) if reliable else None,
    }


def run(
    dataset_path,
    num_speakers=1,
    num_segments=1,
    events_path="./audio/office_events.wav",
    background_path="./audio/ch01ch04-ooffice-demand.wav",
    check_ebr=False,
):
    """Returns (n_checked, n_failed)."""
    dataset_root = Path(dataset_path)
    if not (dataset_root / "ebr-none-sar-none").exists():
        raise FileNotFoundError(
            f"No ebr-none-sar-none/ reference folder found under {dataset_root} -- wrong dataset_path?"
        )

    ids = first_n_ids(dataset_root, num_speakers)
    print(f"Checking {len(ids)} speaker/pair id(s) under {dataset_root}: {ids}\n")

    events_full = events_mono = bg_full = bg_mono = None
    n_checked = 0
    n_failed = 0

    for speaker_id in ids:
        clean_files = first_m_clean_files(dataset_root, speaker_id, num_segments)
        for clean_path in clean_files:
            print(f"=== {speaker_id} / {clean_path.name} ===")
            clean_audio, sr = load_channel_first(clean_path)

            if check_ebr and events_full is None:
                events_full, events_mono = load_source_mono(events_path, sr)
                bg_full, bg_mono = load_source_mono(background_path, sr)

            speech_laeq = compute_LAeq(clean_audio, sr)
            speech_ok = abs(speech_laeq - TARGET_SPEECH_LAEQ) <= SAR_TOLERANCE_DB
            print(f"  clean speech LAeq = {speech_laeq:.2f} dBA (expected ~{TARGET_SPEECH_LAEQ} dBA) "
                  f"[{'OK' if speech_ok else 'UNEXPECTED'}]")

            for ebr_key, sar_key in CONDITIONS:
                noisy_path = condition_path(clean_path, ebr_key, sar_key)
                n_checked += 1
                if not noisy_path.exists():
                    print(f"  ebr-{ebr_key}-sar-{sar_key}: MISSING FILE ({noisy_path})")
                    n_failed += 1
                    continue

                noisy_audio, sr2 = load_channel_first(noisy_path)
                if sr2 != sr or noisy_audio.shape != clean_audio.shape:
                    print(f"  ebr-{ebr_key}-sar-{sar_key}: SHAPE/SR MISMATCH "
                          f"(clean {clean_audio.shape}@{sr} vs noisy {noisy_audio.shape}@{sr2})")
                    n_failed += 1
                    continue

                sar_result = check_sar(clean_audio, noisy_audio, sr, sar_key)
                status = "PASS" if sar_result["pass"] else "FAIL"
                if not sar_result["pass"]:
                    n_failed += 1
                line = (f"  ebr-{ebr_key}-sar-{sar_key}: SAR measured={sar_result['measured_sar']:.2f}dB "
                        f"expected={sar_result['expected_sar']:.2f}dB diff={sar_result['diff']:+.2f}dB [{status}]")

                if check_ebr:
                    y_amb_recovered = noisy_audio - clean_audio
                    ebr_result = evaluate_ebr(
                        y_amb_recovered, sr, events_full, events_mono, bg_full, bg_mono, ebr_key
                    )
                    if ebr_result["pass"] is None:
                        ebr_status = "INCONCLUSIVE"
                    else:
                        ebr_status = "PASS" if ebr_result["pass"] else "FAIL"
                        if not ebr_result["pass"]:
                            n_failed += 1
                    if "measured_ebr" in ebr_result:
                        line += (f" | EBR measured={ebr_result['measured_ebr']:.2f}dB "
                                 f"expected={ebr_result['expected_ebr']:.2f}dB diff={ebr_result['diff']:+.2f}dB "
                                 f"(conf events={ebr_result['conf_events']:.1f} bg={ebr_result['conf_bg']:.1f} "
                                 f"r2={ebr_result['fit_r2']:.2f} events_incr_r2={ebr_result['incremental_r2']:.1e}) [{ebr_status}]")
                    else:
                        line += f" | EBR: {ebr_result['reason']} [INCONCLUSIVE]"

                print(line)
            print()

    print(f"Checked {n_checked} condition files, {n_failed} failed/missing.")
    return n_checked, n_failed
