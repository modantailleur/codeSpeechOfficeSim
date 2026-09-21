"""Copy the VCTK corpus and concatenate utterances into the length-matched groups
produced by the multispeaker-lengths step (./data/matchedgroups.npy).

Ported from the original reformat_vctk_multispeaker_newfolder.py script.
"""
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio_ops import apply_fade, load_mono_audio, load_txt, make_silence, normalize_laeq, save_txt

SILENCE_DUR = 0.1
TARGET_LAEQ = -130  # dBA


def _load_audio(path):
    audio, sr = load_mono_audio(path)
    audio = normalize_laeq(audio, sr, TARGET_LAEQ, source_name=str(path))
    return audio, sr


def _copy_dataset(input_root, output_root, skip_roots):
    print("\n📦 Copying dataset...")

    all_files = list(input_root.rglob("*"))
    total_files = len(all_files)

    for i, src in enumerate(all_files):
        if any(src == root or root in src.parents for root in skip_roots):
            continue

        dst = output_root / src.relative_to(input_root)

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        if i % 500 == 0 or i == total_files - 1:
            print(f"   ➜ Copy progress: {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")

    print("✅ Dataset copy complete\n")


def run(groups_path="./data/matchedgroups.npy",
        input_vctk_dir="../VCTK-Corpus-0.92",
        output_vctk_dir="../VCTK-Corpus-0.92-GROUPED"):
    groups = np.load(groups_path, allow_pickle=True)
    num_groups = len(groups)

    print(f"📦 Loaded {num_groups} groups")

    input_root = Path(input_vctk_dir)
    vctk_txt_root = input_root / "txt"
    vctk_flac_root = input_root / "wav48_silence_trimmed"

    output_root = Path(output_vctk_dir)
    output_txt_root = output_root / "txt"
    output_flac_root = output_root / "wav48_silence_trimmed"

    _copy_dataset(input_root, output_root, skip_roots=(vctk_txt_root, vctk_flac_root))

    # =========================================================
    # PROCESS GROUPS
    # =========================================================
    concat_id = 0
    missing_audio = 0
    missing_txt = 0

    print(f"🔄 Processing {num_groups} groups...\n")

    for gi, group in enumerate(groups):
        group = list(group)

        buffer_audio = []
        buffer_texts = []
        buffer_names = []

        sr_ref = None

        print(f"▶ Group {gi+1}/{num_groups} ({len(group)} items)")

        for item in group:
            base = item.replace(".txt", "")
            speaker = base.split("_")[0]

            audio_path = vctk_flac_root / speaker / f"{base}_mic1.flac"
            txt_path = vctk_txt_root / speaker / f"{base}.txt"

            # ---------------- AUDIO ----------------
            if not audio_path.exists():
                print(f"   ⚠️ Missing audio: {audio_path}")
                missing_audio += 1
                continue

            audio, sr = _load_audio(audio_path)

            sr_ref = sr if sr_ref is None else sr
            audio = apply_fade(audio, sr, fade_dur=0.01)

            if buffer_audio:
                buffer_audio.append(make_silence(sr, SILENCE_DUR))

            buffer_audio.append(audio)
            buffer_names.append(base)

            # ---------------- TEXT ----------------
            if txt_path.exists():
                buffer_texts.append(load_txt(txt_path))
            else:
                print(f"   ⚠️ Missing txt: {txt_path}")
                buffer_texts.append("")
                missing_txt += 1

        if not buffer_audio:
            print("   ⛔ Empty group skipped\n")
            continue

        # =========================================================
        # BUILD OUTPUT NAME
        # =========================================================
        out_audio = np.concatenate(buffer_audio)

        speaker = buffer_names[0].split("_")[0]
        utter_ids = [n.split("_")[1] for n in buffer_names]

        num_str = utter_ids[0] if len(utter_ids) == 1 else f"{utter_ids[0]}-{utter_ids[-1]}"
        concat_base = f"{speaker}_{num_str}"[:200]
        concat_name = f"{concat_base}_mic1"[:200]

        # =========================================================
        # SAVE AUDIO
        # =========================================================
        out_audio_path = output_flac_root / speaker / f"{concat_name}.flac"
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(out_audio_path, out_audio, sr_ref)

        # =========================================================
        # SAVE TEXT
        # =========================================================
        out_txt_path = output_txt_root / speaker / f"{concat_base}.txt"
        save_txt(out_txt_path, " ".join(t for t in buffer_texts if t))

        concat_id += 1

        # progress update
        if gi % 10 == 0 or gi == num_groups - 1:
            print(f"   ✔ Done groups: {gi+1}/{num_groups} ({(gi+1)/num_groups*100:.1f}%)\n")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n================ SUMMARY ================")
    print(f"✅ Grouped files created: {concat_id}")
    print(f"⚠️ Missing audio files: {missing_audio}")
    print(f"⚠️ Missing txt files: {missing_txt}")
    print("========================================\n")
