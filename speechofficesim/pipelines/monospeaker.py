"""Group VCTK per-speaker utterances into ~1 minute monospeaker files.

Ported from the original reformat_vctk_monospeaker.py script.
"""
import os
import shutil

import numpy as np
import soundfile as sf

from ..audio_ops import (
    audio_duration,
    load_mono_audio,
    load_txt,
    make_silence,
    normalize_laeq,
    pad_to_length_centered,
    save_txt,
)

MIN_LEN = 45.0  # seconds
MAX_LEN = 60.0  # seconds
SILENCE_DUR = 0.1  # seconds
TARGET_LAEQ = -130  # dBA


def _load_audio(path):
    audio, sr = load_mono_audio(path)
    audio = normalize_laeq(audio, sr, TARGET_LAEQ, source_name=path)
    return audio, sr


def _reformat_speaker(speaker_name, input_vctk_dir, output_vctk_dir):
    input_audio_dir = os.path.join(input_vctk_dir, "wav48_silence_trimmed", speaker_name)
    input_txt_dir = os.path.join(input_vctk_dir, "txt", speaker_name)
    output_audio_dir = os.path.join(output_vctk_dir, "wav48_silence_trimmed", speaker_name)
    output_txt_dir = os.path.join(output_vctk_dir, "txt", speaker_name)

    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    # ----------------------------
    # 1. Gather mic1 files and split short vs long
    # ----------------------------
    files = sorted(f for f in os.listdir(input_audio_dir) if f.endswith("_mic1.flac"))

    short_files = []
    long_files = []

    for f in files:
        path = os.path.join(input_audio_dir, f)
        audio, sr = _load_audio(path)
        dur = audio_duration(audio, sr)

        entry = {"name": os.path.splitext(f)[0], "audio": audio, "sr": sr, "dur": dur}
        (short_files if dur < MIN_LEN else long_files).append(entry)

    print(f"Found {len(short_files)} short files and {len(long_files)} long files out of {len(files)} total mic1 files.")

    # ----------------------------
    # 2. Save long files as-is (audio + transcript)
    # ----------------------------
    for f in long_files:
        audio_out_path = os.path.join(output_audio_dir, f"{f['name']}.flac")
        sf.write(audio_out_path, f["audio"], f["sr"])

        # Strip "_mic1" from the audio basename to find transcript
        transcript_name = f["name"].replace("_mic1", "")
        txt_in_path = os.path.join(input_txt_dir, f"{transcript_name}.txt")
        txt_out_path = os.path.join(output_txt_dir, f"{transcript_name}.txt")  # keep output name same as audio
        if os.path.exists(txt_in_path):
            shutil.copy(txt_in_path, txt_out_path)
        else:
            print(f"⚠️ Transcript not found for long file: {f['name']}")

    # ----------------------------
    # 3. Greedy concatenation of short files (audio + transcripts)
    # ----------------------------
    pool = short_files.copy()  # remaining files to process
    concat_id = 0
    saved_short_names = []
    discarded_count = 0
    discarded_duration = 0.0

    while pool:
        buffer_audio = []
        buffer_names = []
        buffer_len = 0.0
        buffer_transcripts = []

        i = 0
        while i < len(pool):
            f = pool[i]
            audio = f["audio"]
            dur = f["dur"]

            silence = make_silence(f["sr"], SILENCE_DUR)

            # predicted buffer length if added
            added_len = dur
            if buffer_audio:
                added_len += SILENCE_DUR

            if buffer_len + added_len <= MAX_LEN:
                # append silence if not first clip
                if buffer_audio:
                    buffer_audio.append(silence)
                    buffer_len += SILENCE_DUR

                buffer_audio.append(audio)
                buffer_names.append(f["name"])
                buffer_len += dur

                # read transcript and add (strip _mic1)
                transcript_name = f["name"].replace("_mic1", "")
                txt_path = os.path.join(input_txt_dir, f"{transcript_name}.txt")
                if os.path.exists(txt_path):
                    txt_content = load_txt(txt_path)
                else:
                    print(f"⚠️ Transcript not found for short file: {f['name']}")
                    txt_content = ""
                buffer_transcripts.append(txt_content)

                # remove from pool (we used it)
                pool.pop(i)
            else:
                # cannot add → skip to next in pool
                i += 1

        # Discard the buffer if it doesn't have enough actual speech (e.g. the
        # last leftover buffer, which would otherwise be mostly silence padding)
        if buffer_audio and buffer_len < MIN_LEN:
            print(f"⚠️ Discarding buffer with only {buffer_len:.1f}s of speech (< {MIN_LEN}s min): {buffer_names}")
            discarded_count += len(buffer_names)
            discarded_duration += buffer_len
            buffer_audio = []

        # Save current buffer if not empty
        if buffer_audio:
            out_audio = np.concatenate(buffer_audio)
            out_audio = pad_to_length_centered(out_audio, f["sr"], MAX_LEN)

            # ----------------------------
            # Build new filename in range style
            # ----------------------------
            speaker = buffer_names[0].split("_")[0]  # e.g., "p225"
            # extract numeric part of each utterance
            utter_numbers = [name.split("_")[1] for name in buffer_names]
            num_str = utter_numbers[0] if len(utter_numbers) == 1 else f"{utter_numbers[0]}-{utter_numbers[-1]}"
            concat_name = f"{speaker}_{num_str}_mic1"[:200]

            audio_out_path = os.path.join(output_audio_dir, f"{concat_name}.flac")
            sf.write(audio_out_path, out_audio, f["sr"])

            # save concatenated transcript on a single line
            concat_name_no_mic = concat_name.replace("_mic1", "")
            transcript_out_path = os.path.join(output_txt_dir, f"{concat_name_no_mic}.txt")
            combined_transcript = " ".join(t.strip() for t in buffer_transcripts if t.strip())
            save_txt(transcript_out_path, combined_transcript)

            saved_short_names.extend(buffer_names)
            concat_id += 1

    print(f"✅ Done. Created {concat_id} concatenated short-file buffers with transcripts in {output_audio_dir} / {output_txt_dir} "
          f"({discarded_count} short files discarded for insufficient speech, totaling {discarded_duration:.1f}s of speech lost)")

    # ----------------------------
    # 4. Audit
    # ----------------------------
    all_file_names = [os.path.splitext(f)[0] for f in files]
    all_used_files = [f["name"] for f in long_files] + saved_short_names
    unused_files = set(all_file_names) - set(all_used_files)

    print(f"Total files: {len(all_file_names)}")
    print(f"Used files (long + concatenated short): {len(all_used_files)}")
    print(f"Unused files (should be 0 if all processed): {len(unused_files)}")
    if unused_files:
        print("⚠️ Unused files:")
        for f in unused_files:
            print(f)


def run(input_vctk_dir="../VCTK-Corpus-0.92", output_vctk_dir="../VCTK-Corpus-0.92-MONOSPEAKER"):
    wav_input_dir = os.path.join(input_vctk_dir, "wav48_silence_trimmed")
    speakers_names = [f for f in os.listdir(wav_input_dir) if os.path.isdir(os.path.join(wav_input_dir, f))]

    os.makedirs(output_vctk_dir, exist_ok=True)
    for meta_file in ("speaker-info.txt", "update.txt"):
        meta_in_path = os.path.join(input_vctk_dir, meta_file)
        meta_out_path = os.path.join(output_vctk_dir, meta_file)
        if os.path.exists(meta_in_path):
            shutil.copy(meta_in_path, meta_out_path)
        else:
            print(f"⚠️ {meta_file} not found in {input_vctk_dir}")

    for speaker_name in speakers_names:
        _reformat_speaker(speaker_name, input_vctk_dir, output_vctk_dir)
