"""Helpers for listing/filtering source speech corpora (LibriSpeech, HiFiTTS, VCTK)."""
import json
import os

import pandas as pd
import soundfile as sf


def get_librispeech_audio_paths(librispeech_path, min_length=6, max_length=10):
    """
    Get paths of .flac files in LibriSpeech with length between min_length and max_length seconds.

    Args:
        librispeech_path: Path to the LibriSpeech directory
        min_length: Minimum length in seconds
        max_length: Maximum length in seconds

    Returns:
        List of file paths that meet the length criteria
    """
    # List to store file names and their lengths
    data = []

    # Traverse the directory and its subfolders
    for root, _, files in os.walk(librispeech_path):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                # Read the audio file to get its length
                with sf.SoundFile(file_path) as audio_file:
                    length_in_seconds = len(audio_file) / audio_file.samplerate
                # Append the file name and length to the list
                data.append([file_path, length_in_seconds])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["fname", "length"])

    # Display the DataFrame
    print(df)

    # Filter rows where length is between 6 and 10 seconds
    filtered_df = df[(df['length'] >= min_length) & (df['length'] <= max_length)]
    print(f"Number of rows with length between {min_length}s and {max_length}s: {len(filtered_df)}")
    return filtered_df['fname'].tolist()


def get_hifitts_audio_paths(hifitts_path, min_length=8, max_length=10):
    """
    Get paths of .wav files in HiFiTTS dataset.

    Args:
        hifitts_path: Path to the HiFiTTS directory

    Returns:
        List of file paths
    """

    file_paths = [
        f"{hifitts_path}/92_manifest_clean_train.json",
        f"{hifitts_path}/6097_manifest_clean_train.json",
        f"{hifitts_path}/9017_manifest_clean_train.json",
    ]

    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    data = [item for item in data if min_length <= float(item['duration']) <= max_length]

    for item in data:
        speaker_id = item["audio_filepath"].split("/")[1].split("_")[0]  # Extract speaker ID from path
        item['speaker_id'] = speaker_id
        item['audio_filepath'] = hifitts_path + "/" + item['audio_filepath']

    return data


def get_vctk_audio_paths(vctk_path, min_length=8, max_length=10, only_mic1=False):
    """
    Get paths of .wav files in VCTK dataset.

    Args:
        vctk_path: Path to the VCTK directory

    Returns:
        List of file paths
    """

    audio_file_paths = vctk_path + "/" + "wav48_silence_trimmed"

    data = []
    for root, _, files in os.walk(audio_file_paths):
        for file in files:
            if file.endswith(".flac"):
                if only_mic1 and "mic1" not in file:  # Only consider mic1 recordings if specified
                    continue
                file_path = os.path.join(root, file)
                # Read the audio file to get its length
                with sf.SoundFile(file_path) as audio_file:
                    length_in_seconds = len(audio_file) / audio_file.samplerate
                if min_length <= length_in_seconds <= max_length:
                        speaker_id = file.split("_")[0]
                        file_id = file.split("_")[1]
                        try:
                            text = open(f"{vctk_path}/txt/{speaker_id}/{speaker_id}_{file_id}.txt").read().strip()
                        except FileNotFoundError:
                            continue
                        turns_filepath = f"{vctk_path}/turns/{speaker_id}/{speaker_id}_{file_id}.csv"
                        if not os.path.exists(turns_filepath):
                            turns_filepath = None
                        data.append({"audio_filepath": file_path, "speaker_id": speaker_id, "text_normalized": text, "turns_filepath": turns_filepath})

    print(len(data))
    return data
