# Speech Office Dataset: A simulated dataset of speech in office

## Introduction

This is the code used to reproduce the [Speech Office Dataset](https://zenodo.org/records/22687159). Please refer to the Zenodo repository for more information about the dataset itself.

This repo mixes 3 type of audio files:
- speech
- events
- background

Speech is taken from the VCTK dataset, available at: https://www.kaggle.com/datasets/pratt3000/vctk-corpus 

Events are taken from 2 audios from freesound, which are combined together in ./audio/office_events.wav . It contains keyboard taping, and some events happening in an office (microwave, people coughing, etc...):
- Keyboard typing in office: https://freesound.org/people/bonnyorbit/sounds/399823/
- Office ambience: https://freesound.org/people/chelly01/sounds/541117/

Background is taken from ch01 and ch04 of the file ooffice of the demand dataset, available at:
https://www.kaggle.com/datasets/chrisfilo/demand .

From that, we create 2 datasets:
- monospeaker (**SOS-1SP**): we mix 1min of speech (concatenated files) from a single person, and 1 consecutive min of events and backgrounds randomly sampled.
- multispeaker (**SOS-2SP**): we make an analysis of the CallHome corpus (available at: https://talkbank.org/ca/access/CallHome/eng.html), to find what's the distribution of the length in words of a person speaking in an english conversation, before someone else talks. We then concatenate randomly segments of audio of an individual speaker, and segments of audio of another speaker based on that. We create couples of speakers, meaning that we associate all data of 2 speakers together, and we never use them for conversation with other speakers of the dataset.

## Repository layout

```
speechofficesim/            # the package: `python -m speechofficesim <subcommand>`
├── cli.py                  # argparse subcommands, one per pipeline step below
├── acoustics.py            # A-weighting / LAeq / room acoustics simulation
├── datasets.py             # LibriSpeech / HiFiTTS / VCTK path & metadata helpers
├── audio_ops.py            # shared audio & transcript I/O helpers
└── pipelines/               # one module per pipeline step (see table below)
tests/                      # pytest suite covering the modules above
audio/                       # bundled sample clips (events, background, calibration)
data/                        # cached intermediate arrays for the multispeaker pipeline (gitignored)
dataset_CABank English CallHome Corpus/   # CallHome transcripts (external, gitignored)
```

## Setup

First, install the required dependencies in a new **Python 3.9.15** environment:

```
pip install -r requirements.txt
```

All commands below are run from the repository root with `python -m speechofficesim <subcommand>`; no installation of the package itself is required. Every subcommand also accepts `--help` to list its flags and defaults.

All the paths below are relative to the repository root and default to `../`, i.e. one directory above this repo (a sibling of `codeSpeechOfficeSim/`). Every command also accepts flags to point elsewhere — run it with `--help` to see them.

### Getting VCTK

Both SOS-1SP and SOS-2SP are built from VCTK Corpus 0.92. Download it from the official University of Edinburgh DataShare page:

https://datashare.ed.ac.uk/handle/10283/3443

(a mirror is also available on Kaggle: https://www.kaggle.com/datasets/pratt3000/vctk-corpus)

Extract it so it sits as a sibling of this repo, named `VCTK-Corpus-0.92`:

```
post-doc/Code/
├── codeSpeechOfficeSim/          <- this repo
└── VCTK-Corpus-0.92/             <- extracted VCTK, with wav48_silence_trimmed/, txt/, speaker-info.txt, ...
```

That's the `../VCTK-Corpus-0.92` default every command below expects for its `--input_vctk_dir`/`--vctk_dir`/`--vctk_path` flag. If you'd rather keep VCTK somewhere else, just pass the matching flag on each command.

## SOS-1SP dataset

To group the vctk files into 1min monospeaker files:

```
python -m speechofficesim monospeaker-reformat
# -> ../VCTK-Corpus-0.92-MONOSPEAKER/
```

Then you can use the following code to mix those speech files with events and background:

```
python -m speechofficesim mix
# -> ../SOS-1SP/
```

## SOS-2SP dataset

To save the files of VCTK that should be grouped together for a single speaker:

```
python -m speechofficesim multispeaker-lengths
# -> ./data/matchedgroups.npy (and callhome.npy / vctk.npy / matched.npy)
```

This will create the file "./data/matchedgroups.npy", which is then used to actually group the files in the code:

```
python -m speechofficesim multispeaker-group
# -> ../VCTK-Corpus-0.92-GROUPED/
```

You can then create the conversational vctk dataset of 1min files, by launching:

```
python -m speechofficesim multispeaker-conversations
# -> ../VCTK-Corpus-0.92-CONVERSATIONS/
```

Then you can use the following code to mix those speech files with events and background:

```
python -m speechofficesim mix --vctk_path "../VCTK-Corpus-0.92-CONVERSATIONS" --out_dataset_path "../SOS-2SP/"
# -> ../SOS-2SP/
```

## Ground-truth VAD annotations

To generate groundtruth annotations for speech presence, point `--dataset_path` at the mixed dataset you want to annotate (its default matches the monospeaker output above):

```
# Monospeaker
python -m speechofficesim vad --dataset_path "../SOS-1SP"
# -> ../SOS-1SP/vadgt/

# Multispeaker
python -m speechofficesim vad --dataset_path "../SOS-2SP"
# -> ../SOS-2SP/vadgt/
```
