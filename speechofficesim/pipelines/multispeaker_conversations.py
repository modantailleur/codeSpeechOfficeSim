"""Build ~1 minute two-speaker conversations out of the grouped VCTK corpus.

Ported from the original reformat_vctk_multispeaker_newfolder_conversations.py
script.
"""
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio_ops import load_mono_audio, load_txt, make_silence, save_txt

SILENCE_DUR = 0.5   # gap between turns
TARGET_DUR = 60.0   # 1 minute file
LOWER_DUR = 0.8 * TARGET_DUR   # 80% of target
UPPER_DUR = TARGET_DUR         # going above this triggers a redraw
MAX_TRIES = 1000
MAX_ITEM_DUR = 20.0  # discard original audios longer than this


def _save_turns_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "text", "timestamp", "original_filename"])
        writer.writerows(rows)


def _word_count(text):
    return len(text.split())


def _load_speaker_pool(speaker, input_txt_root, input_flac_root):
    txt_dir = input_txt_root / speaker
    flac_dir = input_flac_root / speaker

    items = []
    for txt_path in sorted(txt_dir.glob("*.txt")):
        base = txt_path.stem
        flac_path = flac_dir / f"{base}_mic1.flac"
        if not flac_path.exists():
            continue

        text = load_txt(txt_path)
        info = sf.info(flac_path)
        duration = info.frames / info.samplerate

        if duration > MAX_ITEM_DUR:
            continue

        items.append({
            "base": base,
            "txt_path": txt_path,
            "flac_path": flac_path,
            "text": text,
            "words": _word_count(text),
            "duration": duration,
        })

    return items


def _try_build_conversation(pools, spk1, spk2):
    items1 = pools[spk1][:]
    items2 = pools[spk2][:]
    random.shuffle(items1)
    random.shuffle(items2)

    turns = []
    total_dur = 0.0
    idx1 = idx2 = 0
    turn_is_spk1 = True

    while True:
        current_items, idx = (items1, idx1) if turn_is_spk1 else (items2, idx2)

        if idx >= len(current_items):
            break

        item = current_items[idx]
        if turn_is_spk1:
            idx1 += 1
        else:
            idx2 += 1

        added_dur = item["duration"] + (SILENCE_DUR if turns else 0.0)

        if total_dur + added_dur > UPPER_DUR:
            break

        turns.append((spk1 if turn_is_spk1 else spk2, item))
        total_dur += added_dur

        if total_dur >= LOWER_DUR:
            break

        turn_is_spk1 = not turn_is_spk1

    valid = LOWER_DUR <= total_dur <= UPPER_DUR
    return valid, turns, total_dur


def _save_conversation(conv_id, spk1, spk2, turns, output_flac_root, output_txt_root, output_turns_root):
    buffer_audio = []
    buffer_lines = []
    turn_rows = []
    sr_ref = None
    elapsed = 0.0

    for speaker, item in turns:
        audio, sr = load_mono_audio(item["flac_path"])
        sr_ref = sr if sr_ref is None else sr

        if buffer_audio:
            buffer_audio.append(make_silence(sr, SILENCE_DUR))
            elapsed += SILENCE_DUR

        buffer_audio.append(audio)
        buffer_lines.append(f"[{speaker}] {item['text']}")
        turn_rows.append((speaker, item["text"], f"{elapsed:.3f}", item["base"]))

        elapsed += len(audio) / sr

    out_audio = np.concatenate(buffer_audio)

    target_len = int(round(TARGET_DUR * sr_ref))
    pad_total = max(0, target_len - len(out_audio))
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    if pad_total > 0:
        out_audio = np.concatenate([
            make_silence(sr_ref, pad_left / sr_ref),
            out_audio,
            make_silence(sr_ref, pad_right / sr_ref),
        ])

    pad_left_dur = pad_left / sr_ref
    turn_rows = [
        (speaker, text, f"{float(timestamp) + pad_left_dur:.3f}", base)
        for speaker, text, timestamp, base in turn_rows
    ]

    pair_name = "-".join(sorted((spk1, spk2)))
    conv_name = f"{pair_name}_conv{conv_id:04d}"

    out_audio_path = output_flac_root / pair_name / f"{conv_name}_mic1.flac"
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_audio_path, out_audio, sr_ref)

    out_txt_path = output_txt_root / pair_name / f"{conv_name}.txt"
    save_txt(out_txt_path, "\n".join(buffer_lines))

    out_turns_path = output_turns_root / pair_name / f"{conv_name}.csv"
    _save_turns_csv(out_turns_path, turn_rows)

    final_dur = len(out_audio) / sr_ref
    print(f"   ✔ Saved {conv_name} ({final_dur:.1f}s, {len(turns)} turns)")


def _remove_used(pools, spk, used_bases):
    pools[spk] = [item for item in pools[spk] if item["base"] not in used_bases]
    if not pools[spk]:
        del pools[spk]


def run(input_vctk_dir="../VCTK-Corpus-0.92-GROUPED",
        output_vctk_dir="../VCTK-Corpus-0.92-CONVERSATIONS"):
    random.seed(0)
    np.random.seed(0)

    input_root = Path(input_vctk_dir)
    input_txt_root = input_root / "txt"
    input_flac_root = input_root / "wav48_silence_trimmed"

    output_root = Path(output_vctk_dir)
    output_txt_root = output_root / "txt"
    output_turns_root = output_root / "turns"
    output_flac_root = output_root / "wav48_silence_trimmed"

    # =========================================================
    # COPY META FILES
    # =========================================================
    output_root.mkdir(parents=True, exist_ok=True)
    for meta_file in ("speaker-info.txt", "update.txt"):
        meta_in_path = input_root / meta_file
        meta_out_path = output_root / meta_file
        if meta_in_path.exists():
            shutil.copy(meta_in_path, meta_out_path)
        else:
            print(f"⚠️ {meta_file} not found in {input_root}")

    print("📦 Loading grouped speaker pools...")

    speakers = sorted(d.name for d in input_txt_root.iterdir() if d.is_dir())

    pools = {spk: _load_speaker_pool(spk, input_txt_root, input_flac_root) for spk in speakers}
    pools = {spk: items for spk, items in pools.items() if items}

    total_input_dur = sum(item["duration"] for items in pools.values() for item in items)

    print(f"✅ Loaded {len(pools)} speakers with remaining audio\n")

    # =========================================================
    # MAIN LOOP
    # =========================================================
    print("🔄 Building conversations...\n")

    sorted_speakers = sorted(pools.keys())
    speaker_pairs = [
        (sorted_speakers[i], sorted_speakers[i + 1])
        for i in range(0, len(sorted_speakers) - 1, 2)
    ]

    conv_id = 0

    for spk1, spk2 in speaker_pairs:
        if spk1 not in pools or spk2 not in pools:
            continue

        print(f"👥 Pairing {spk1}/{spk2}\n")

        while spk1 in pools and spk2 in pools:
            success = False

            for _attempt in range(MAX_TRIES):
                valid, turns, total_dur = _try_build_conversation(pools, spk1, spk2)

                if valid:
                    conv_id += 1
                    used1 = {item["base"] for s, item in turns if s == spk1}
                    used2 = {item["base"] for s, item in turns if s == spk2}

                    _save_conversation(conv_id, spk1, spk2, turns, output_flac_root, output_txt_root, output_turns_root)

                    _remove_used(pools, spk1, used1)
                    _remove_used(pools, spk2, used2)

                    success = True
                    break

            if not success:
                print(f"   ⚠️ Could not build another valid conversation for {spk1}/{spk2} "
                      f"after {MAX_TRIES} tries — moving to next pair\n")
                break

    print("⛔ No more speaker pairs left to process.\n")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n================ SUMMARY ================")
    leftover_dur = sum(item["duration"] for items in pools.values() for item in items)
    unused_pct = 100 * leftover_dur / total_input_dur if total_input_dur else 0.0

    print(f"✅ Conversations created: {conv_id}")
    print(f"🗑️  Unused original audio: {unused_pct:.1f}% ({leftover_dur / 60:.1f} min of {total_input_dur / 60:.1f} min)")
    print("========================================\n")
