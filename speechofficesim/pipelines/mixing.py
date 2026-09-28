"""Mix VCTK speech (mono- or multi-speaker) with office events and background noise.

Ported from the original mix_speech_events_background.py script.
"""
import os
import shutil

import librosa
import numpy as np
import soundfile as sf

from ..acoustics import RoomAcousticProcessor, compute_LAeq
from ..datasets import get_vctk_audio_paths

EBRS = {
    "low": -50,  # Event is -50dB louder than background
    "mid": 3,   # Event is 3dB louder than background
    "high": 12,   # Event is 12dB louder than background
}

SARS = {
    "low": 20,  # Ambient is 20dB quieter than speech
    "mid": 10,  # Ambient is 10dB quieter than speech
    "high": 0,  # Ambient is 0dB quieter than speech
}

# Reference level the room-processed speech is leveled to before any ambient
# content is added (see the derivation in run(), below). Exposed as a module
# constant so other tools (e.g. pipelines/integrity_check.py) can check
# against it without duplicating the number.
TARGET_SPEECH_LAEQ = -136.93  # dBA


def mix_scene(y_speech_scaled, y_events, y_bg, sr, ebr="mid", sar="mid"):
    """
    Mix speech, background noise, and event sounds into a single audio signal with controlled levels.

    This function combines three audio components (speech, background noise, and events) by scaling
    them according to specified acoustic ratios. The mixing process occurs in two stages:
    1. Events are scaled relative to background noise using the Event-to-Background Ratio (EBR)
    2. The combined ambient sound (events + background) is scaled relative to speech using
        the Speech-to-Ambient Ratio (SAR)

    Args:
         y_speech_scaled (numpy.ndarray): Pre-scaled speech signal.
         y_events (numpy.ndarray): Event sounds (e.g., keyboard typing, door closing).
         y_bg (numpy.ndarray): Background noise signal.
         sr (int): Sample rate in Hz for all audio signals.
         ebr (str, optional): Event-to-Background Ratio key. Determines the relative level of
              events compared to background noise. Default is "mid".
         sar (str, optional): Speech-to-Ambient Ratio key. Determines the relative level of
              speech compared to the combined ambient sound. Default is "mid".

    Returns:
         numpy.ndarray: Mixed audio signal with all three components combined at the specified levels.

    Notes:
         - All input signals must have the same sample rate.
         - The function uses A-weighted equivalent continuous sound level (LAeq) for scaling calculations.
         - ebr and sar parameters should correspond to keys in global dictionaries 'EBRS' and 'SARS'.
    """
    laeq_speech = compute_LAeq(y_speech_scaled, sr)
    laeq_events = compute_LAeq(y_events, sr)
    laeq_bg = compute_LAeq(y_bg, sr)

    # Scale events to match target level relative to speech (in dBA). This is
    # the EBR (Event-to-Background Ratio) in dB.
    target_events_laeq = laeq_bg + EBRS[ebr]
    scale_events = 10 ** ((target_events_laeq - laeq_events) / 20)
    y_events_scaled = y_events * scale_events

    # Ambient is the mix of events and background
    y_amb = y_events_scaled + y_bg
    laeq_amb = compute_LAeq(y_amb, sr)

    # Scale ambient to match target level relative to speech (in dBA). This is
    # the SAR (Speech-To-Ambient Ratio) in dB.
    target_amb_laeq = laeq_speech - SARS[sar]
    scale_amb = 10 ** ((target_amb_laeq - laeq_amb) / 20)
    y_amb_scaled = y_amb * scale_amb

    y_mixed = y_speech_scaled + y_amb_scaled
    return y_mixed


def run(
    sr=48000,
    events_path="./audio/office_events.wav",  # Mix of keyboard from: ... and office ambien from: ...
    background_path="./audio/ch01ch04-ooffice-demand.wav",  # Taken from: https://freesound.org/people/mzui/sounds/203306/
    segment_duration=60,  # seconds
    vctk_path="../VCTK-Corpus-0.92-MONOSPEAKER",
    out_dataset_path="../SOS-1SP/",
):
    # Set random seed for reproducibility
    np.random.seed(0)

    # -----------------------------
    # Load files
    # -----------------------------
    y_events, _ = librosa.load(events_path, sr=sr, mono=False)
    y_bg, _ = librosa.load(background_path, sr=sr, mono=False)

    segment_samples = segment_duration * sr

    # Randomly select start positions
    max_start_events = y_events.shape[1] - segment_samples
    max_start_bg = y_bg.shape[1] - segment_samples

    speech_audio_paths = get_vctk_audio_paths(vctk_path=vctk_path, min_length=int(segment_duration * 0.5), max_length=segment_duration)

    os.makedirs(out_dataset_path, exist_ok=True)

    total_iterations = len(speech_audio_paths) * (len(EBRS.keys()) * len(SARS.keys()) + 1) * len([0])  # Only pan=0 for now; +1 for the none/none case
    iteration = 0

    for speech_path in speech_audio_paths:
        for pan in [0]:  # Pan from center to right
            for ebr in list(EBRS.keys()) + ["none"]:
                for sar in list(SARS.keys()) + ["none"]:
                    if (ebr == "none" and sar != "none") or (ebr != "none" and sar == "none"):
                        continue
                    iteration += 1  # increment counter
                    speech_fname = os.path.basename(speech_path["audio_filepath"]).split('.')[0]

                    # according to a preliminary study, with
                    # our LAeq computation, the office ambience mixed with keyboard typing is
                    # at -131.37dBA.
                    # - according to this paper : https://arxiv.org/pdf/2305.01762
                    # The mean Laeq range in an office is 53.56dB
                    # - according to this paper : Speech level variation by office environment and communication type
                    # The findings reveal an average speech level of 54.0 dBA at 1 m, so it would be 54,0dB-6dB=48dB at 2m
                    #
                    # So the relative level between the office and speech
                    # background should be of 5.56dB.
                    # So we need to attenuate the speech by 6dB to match the office ambience
                    # So on our Compute LAeq, we need to set the speech level to around -136,93dBA

                    y_speech, sr = librosa.load(speech_path["audio_filepath"], sr=sr)
                    y_speech = y_speech[:segment_samples]
                    signal = y_speech / (np.max(np.abs(y_speech)) + 1e-9)

                    room_processor = RoomAcousticProcessor(sr)
                    y_speech_p = room_processor.process(signal, pan=pan)

                    # Ensure correct length: truncate if too long, zero-pad if too short
                    if y_speech_p.shape[1] > segment_samples:
                        y_speech_p = y_speech_p[:, :segment_samples]
                    elif y_speech_p.shape[1] < segment_samples:
                        pad_total = segment_samples - y_speech_p.shape[1]
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        y_speech_p = np.pad(y_speech_p, ((0, 0), (pad_left, pad_right)), mode='constant')

                    # Stereo output -> compute average across channels
                    y_speech_p_dbA = compute_LAeq(y_speech_p, sr)

                    # Compute linear scale factor
                    scale = 10 ** ((TARGET_SPEECH_LAEQ - y_speech_p_dbA) / 20)

                    # Apply to speech signal
                    y_speech_scaled = y_speech_p * scale

                    #####################
                    # Sample in office and keyboard audio
                    if ebr == "none" and sar == "none":
                        y_mixed = y_speech_scaled
                    else:
                        start_events = np.random.randint(0, max_start_events)
                        start_bg = np.random.randint(0, max_start_bg)

                        # Extract segments
                        y_events_seg = y_events[:, start_events:start_events + segment_samples]
                        y_bg_seg = y_bg[:, start_bg:start_bg + segment_samples]

                        y_mixed = mix_scene(y_speech_scaled, y_events_seg, y_bg_seg, sr, ebr=ebr, sar=sar)

                    #################################
                    # Mix speech with office ambience
                    output_dir = f"{out_dataset_path}/ebr-{ebr}-sar-{sar}/pan_{int(pan*100)}/"
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(f"{output_dir}/{speech_path['speaker_id']}", exist_ok=True)
                    output_path = f"{output_dir}/{speech_path['speaker_id']}/spk_{speech_path['speaker_id']}__{speech_fname}__pan_{int(pan*100)}__ebr-{ebr}-sar-{sar}.flac"

                    sf.write(output_path, y_mixed.T, sr, format="FLAC", subtype="PCM_16")

                    print(f"[{iteration}/{total_iterations}] Processed {output_path}")

                    transc_dir = f"{out_dataset_path}/transc/"
                    os.makedirs(transc_dir, exist_ok=True)
                    # Store text_normalize field to a .txt file
                    text_file_path = f"{transc_dir}/{speech_fname}.txt"
                    if not os.path.exists(text_file_path):
                        with open(text_file_path, 'w') as f:
                            f.write(speech_path["text_normalized"])

                    # Mirror the turns CSV (speaker turns within the 1min conversation)
                    # under a "turns" folder that follows the same hierarchy as the audio.
                    if speech_path.get("turns_filepath"):
                        turns_output_dir = output_dir.replace(f"{out_dataset_path}/", f"{out_dataset_path}/turns/", 1)
                        os.makedirs(f"{turns_output_dir}/{speech_path['speaker_id']}", exist_ok=True)
                        turns_output_path = f"{turns_output_dir}/{speech_path['speaker_id']}/spk_{speech_path['speaker_id']}__{speech_fname}__pan_{int(pan*100)}__ebr-{ebr}-sar-{sar}.csv"
                        shutil.copy2(speech_path["turns_filepath"], turns_output_path)

    # Path to the VCTK speaker info file
    speaker_info_src = os.path.join(vctk_path, "speaker-info.txt")

    # Destination (root of your generated dataset)
    speaker_info_dst = os.path.join(out_dataset_path, "speaker-info.txt")

    # Copy if it exists and not already copied
    if os.path.exists(speaker_info_src):
        shutil.copy2(speaker_info_src, speaker_info_dst)
        print(f"Copied speaker-info.txt to {speaker_info_dst}")
    else:
        print("speaker-info.txt not found in VCTK root")
