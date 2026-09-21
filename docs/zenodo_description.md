# SOS: Simulated Office Speech — SOS-1SP & SOS-2SP

**Modan Tailleur, Théo Chasle Cauchy, Mathieu Lagrange**

**[TODO: affiliations, ORCID iDs]**

**[TODO: DOI badge — filled in automatically by Zenodo on publish]**

## Summary

This record distributes two simulated datasets of speech recorded in an office environment, **SOS-1SP** (single-speaker, monologue-style) and **SOS-2SP** (two-speaker, conversation-style). Both are built by acoustically rendering read speech from the VCTK corpus in a simulated stereo room, then mixing it with real office background noise and office-specific sound events (keyboard typing, ambience) at a set of controlled, reproducible Event-to-Background and Speech-to-Ambient level ratios. SOS-2SP additionally reshapes VCTK's read-speech utterances into two-speaker exchanges whose turn-length statistics are matched to real telephone conversations (CallHome), so that its turn-taking rhythm resembles natural dialogue rather than the isolated, unrelated utterances VCTK speakers originally recorded.

Both datasets are intended for research on speech activity detection, speech enhancement/separation, and automatic speech recognition under realistic, parametrically-controlled office noise conditions, with frame-level ground-truth speech-activity annotations included.

## Motivation

Office environments are one of the most common real-world settings for voice technology (meeting transcription, hands-free assistants, hybrid-work telephony), yet public datasets that combine (a) clean, high-quality speech, (b) *authentic* office background/event recordings, and (c) systematically controlled, labelled mixing conditions are scarce. Existing conversational speech corpora recorded over the telephone (e.g. CallHome) are natural and well-suited for studying turn-taking, but are heavily band-limited and compressed, which makes them unsuitable as a *source* for the clean, controllable audio needed to build parametric noisy-speech benchmarks. VCTK offers exactly that clean audio quality, at the cost of not containing any spontaneous conversational structure — every speaker reads the same fixed set of sentences in isolation, without another speaker present.

SOS-1SP and SOS-2SP were built to close that gap: they keep VCTK's audio quality while (i) rendering it through a simulated office acoustic scene under a systematic grid of noise conditions, and (ii) in the two-speaker case, restructuring it so its conversational timing statistics resemble a real telephone corpus, without needing to redistribute that corpus's actual (restricted-access) audio.

## Dataset overview

| | SOS-1SP | SOS-2SP |
|---|---|---|
| Speaking style | Single speaker, VCTK utterances concatenated back-to-back | Two speakers, alternating turns |
| Turn-length statistics | N/A (continuous single-speaker speech) | Matched to the CallHome English corpus's per-turn word-count distribution |
| Segment length | 1 minute per file | 1 minute per file |
| Noise conditions per source segment | 10 (9 controlled + 1 clean reference) | 10 (9 controlled + 1 clean reference) |
| Per-file ground truth | Frame-level speech-activity annotation | Frame-level speech-activity annotation + per-turn speaker/text/timestamp metadata |

Every 1-minute source segment (a single-speaker file for SOS-1SP, a two-speaker conversation for SOS-2SP) is rendered under all 10 noise conditions described below, so the same underlying speech content is available at every point on the noise-condition grid — useful for controlled comparisons (e.g. evaluating a model's degradation curve as noise conditions get harder).

## Source materials

| Component | Source | Role |
|---|---|---|
| Speech | [VCTK Corpus 0.92](https://datashare.ed.ac.uk/handle/10283/3443) — 111 English speakers reading the same fixed set of newspaper/elicitation sentences | Foreground speech content for both datasets |
| Office events | [Keyboard typing in office](https://freesound.org/people/bonnyorbit/sounds/399823/) (Freesound, bonnyorbit) + [Office ambience](https://freesound.org/people/chelly01/sounds/541117/) (Freesound, chelly01), pre-mixed into a single events bed | Discrete/semi-discrete office sound events (typing, incidental ambience) |
| Background noise | Channels 1 and 4 of the "OOFFICE" recording, [DEMAND](https://www.kaggle.com/datasets/chrisfilo/demand) | Continuous office background noise bed |
| Conversational structure reference | [CallHome English corpus](https://talkbank.org/ca/access/CallHome/eng.html) (via TalkBank) | Statistical reference only — see below |

**Important note on CallHome:** CallHome's own audio and transcripts are access-restricted (TalkBank/LDC terms) and are **not** included in, or redistributed by, this dataset. CallHome is used only as a *reference corpus* during construction, to measure the empirical distribution of how many words a person speaks before their conversation partner takes a turn. That distribution is then used to decide how VCTK utterances (which contain no natural conversational turns) get grouped into turn-length-matched segments for SOS-2SP. No CallHome audio, transcript text, or per-utterance content appears anywhere in the released data.

## Dataset construction

### 1. Single-speaker segments (SOS-1SP)

For each of VCTK's 111 speakers, that speaker's individual utterance recordings are concatenated (short silence gaps between utterances) into contiguous single-speaker segments of roughly one minute, using a simple greedy bin-packing over utterance durations. Recordings that are already close to one minute long are kept as-is; segments that end up too short to be usable (i.e. mostly padding) are dropped. Every source utterance is either used or explicitly logged as discarded, so segment construction is fully accounted for. The result is a corpus of single-speaker, continuously-speaking one-minute files — one speaker "talking to camera" per file, as if reading through material at their desk.

### 2. Two-speaker conversations (SOS-2SP)

Building a natural-feeling two-speaker exchange out of VCTK requires two things VCTK doesn't provide: turn-length variability that matches real conversation, and a way to pair two speakers' utterances into alternating turns. This is done in three steps:

1. **Turn-length reference.** CallHome's transcripts are parsed to extract, for every uninterrupted stretch of speech by one participant before the other takes over, the number of words spoken in that turn. This produces an empirical distribution of "how long is a conversational turn" in real, spontaneous, two-party English telephone conversation.
2. **Length-matched grouping.** Each VCTK speaker's utterances are grouped into "pseudo-turns": for a target turn length sampled from the CallHome distribution, a set of that speaker's own utterances is assembled (via randomized search) whose combined word count falls within ±20% of the target. This is repeated until every VCTK utterance has been assigned to exactly one pseudo-turn group (any utterance that can't be matched to a sampled target becomes a singleton group of its own, so no material is discarded). Each group is then concatenated into a single audio clip (short cross-fades, brief silence gaps between the constituent utterances), giving a pool of "pseudo-turn" clips per speaker whose *duration statistics*, not content, mimic real conversational turns.
3. **Conversation assembly.** Speakers are paired up, and pseudo-turn clips are drawn alternately from each speaker's pool (with a short silence gap between turns) until the accumulated duration falls within 80-100% of the one-minute target, then padded symmetrically with silence to exactly one minute. This repeats, drawing fresh (not-yet-used) pseudo-turns each time, until one of the two speakers in a pair runs out of usable material. Each resulting conversation file has an accompanying per-turn metadata table (speaker id, transcript text, onset timestamp, source VCTK utterance id).

Speaker pairs are fixed for the whole dataset (each VCTK speaker is paired with exactly one partner), so no speaker's voice appears in conversation with more than one other speaker — this keeps within-pair speaker-pair identification a well-posed task and avoids one speaker's data leaking across many different "conversations" in a way that could bias speaker-dependent evaluation.

### 3. Acoustic scene rendering

Each one-minute speech segment (from either step above) is spatialized using a simulated shoebox room (image-source method, 5th-order reflections, with air absorption) representing a moderately absorptive office (10m × 10m × 3m, wall absorption coefficient 0.80). A stereo microphone pair (20 cm spacing, comparable to a spaced pair at a desk or laptop) is placed at the room's center at a seated ear height of 1.5 m; the speaker is placed 2 m from the microphone pair. The current release uses a single, centered source position (directly facing the microphone pair); the simulation code supports azimuth offsets up to 90° for future lateral-position releases, but that is not exercised here — every file in this release corresponds to a centered talker.

The room-rendered speech is then leveled to a fixed reference chosen from acoustics literature: an office's mean ambient level is reported around 53.6 dBA ([Vidal et al., 2023](https://arxiv.org/pdf/2305.01762)), and typical conversational speech level is reported around 54.0 dBA at 1 m (per a separate study on speech level variation by office environment and communication type), which drops to roughly 48 dBA once distance-attenuated to 2 m — implying speech should sit roughly 5.6 dB below the office's own ambient floor at that distance. This relative offset is applied on top of the actual measured level of this dataset's own office-events source clip, giving a fixed internal reference level for the rendered speech before any additional noise is mixed in.

**A note on levels:** all internal "LAeq" values used during construction (including the reference above) are computed with an approximate A-weighting filter on an internal digital full-scale reference, not calibrated against a real-world sound pressure level (dB SPL) reference. They should be read as *relative* levels — consistent and comparable to one another within this dataset — not as absolute acoustic measurements. The literature-derived numbers above were used only to choose a realistic relative gap between speech and ambient noise, not to claim SPL calibration.

### 4. Controlled mixing conditions

Once a segment's speech is rendered and leveled, office events and background noise are added on top of it at nine controlled combinations of two independent ratios, plus one clean reference condition with no added events or background:

- **EBR (Event-to-Background Ratio):** how loud the office sound-event bed (keyboard typing / ambience) is relative to the background noise bed.
- **SAR (Speech-to-Ambient Ratio):** how loud the combined ambient bed (events + background, after EBR scaling) is relative to the already-leveled speech, expressed as how far below speech level the ambient bed is set.

| EBR label | Event level relative to background | Perceptual effect |
|---|---|---|
| `low` | −50 dB (background) | Office events are effectively inaudible / absent — background noise dominates alone |
| `mid` | +12 dB (events) | Office events are clearly audible and prominent above the background |
| `high` | +3 dB (events) | Office events sit only slightly above the background, blending into it |

| SAR label | Ambient level relative to speech | Perceptual effect |
|---|---|---|
| `low` | −20 dB (ambient is 20 dB below speech) | Ambient noise is barely audible; easiest listening condition |
| `mid` | −10 dB (ambient is 10 dB below speech) | Ambient noise is clearly present but subordinate to speech |
| `high` | 0 dB (ambient equals speech level) | Ambient noise is as loud as speech; hardest listening condition |

**Note on the EBR labels:** unlike the SAR labels, which are monotonic in listening difficulty (`low` easiest → `high` hardest), the EBR labels are not ordered by event loudness: `mid` actually produces *more* prominent events than `high` (+12 dB vs. +3 dB relative to background). The labels instead correspond to three qualitatively distinct event-salience regimes (absent / prominent / blended-with-background) rather than a monotonic loudness scale — this is intentional, but worth keeping in mind when filtering or comparing conditions by label alone.

Crossing the two ratios gives 9 mixed conditions (`ebr-{low,mid,high}-sar-{low,mid,high}`), plus a 10th **clean reference condition** (`ebr-none-sar-none`) containing the rendered, leveled speech with *no* events or background added at all — used both as an anechoic-noise-free reference point and as the basis for the ground-truth speech-activity annotations (see below). Events and background segments are drawn from independently, randomly chosen time offsets within their respective source recordings for every rendered file (fixed random seed for full reproducibility), so no two mixed files share identical noise content even within the same condition.

### 5. Reproducibility

All randomized steps (turn-length matching, speaker/utterance shuffling in conversation assembly, event/background segment sampling during mixing) use fixed random seeds, so the entire pipeline is deterministic and exactly reproducible from the source corpora.

## Ground-truth annotations

Frame-level speech-activity (voice activity detection, VAD) ground truth is provided for every segment. It is computed directly on the clean reference condition (`ebr-none-sar-none`): the signal is peak-normalized, split into non-overlapping 50 ms frames, and each frame is labelled active (1) or inactive (0) by thresholding its frame-level RMS level (−36 dB relative to the signal's own peak). Because the underlying rendered speech is identical across all 10 noise conditions for a given segment (only the added events/background differ), a single VAD annotation per segment is valid ground truth for that segment's clean condition *and* all 9 noisy mixes of it — it does not need to be, and is not, recomputed per noise condition.

## Dataset organization

Each dataset (SOS-1SP, SOS-2SP) is organized as:

- One audio file per (segment, noise condition) pair, grouped first by noise condition, then by speaker. File and folder names encode the speaker id, the source segment name, the (currently always centered) pan position, and the EBR/SAR condition.
- A shared transcripts folder, with one text file per source segment (shared across its 10 noise-condition renders), containing the segment's normalized transcript (for SOS-2SP, one file per conversation, with speaker turns marked).
- (SOS-2SP only) A turns metadata folder mirroring the audio hierarchy, with one table per conversation file giving each turn's speaker id, transcript text, onset timestamp within the file, and source VCTK utterance id.
- A ground-truth annotations folder, with one frame-level speech-activity file per source segment (shared across its 10 noise-condition renders, per the note above).
- VCTK's own speaker metadata table (speaker id, age, gender, accent, region) at the root of each dataset, for convenience.

Audio is delivered as FLAC (lossless), stereo, 48 kHz, 16-bit.

There is no predefined train/validation/test split — all segments across all speakers are included, and users are expected to define their own splits (e.g. by speaker, to test speaker-independent generalization) appropriate to their task.

## Dataset statistics

**[TODO: fill in after a full pipeline run]**

| | SOS-1SP | SOS-2SP |
|---|---|---|
| Speakers | [N] | [N] speakers, [N] speaker pairs |
| Source segments (before noise conditions) | [N] | [N] conversations |
| Total files (all noise conditions) | [N] | [N] |
| Total audio duration | [X] hours | [X] hours |
| Mean segment duration | [X] s | [X] s |

## Known limitations

- Rendered speech uses a single, centered talker position; no lateral or off-axis positions are included in this release, though the construction pipeline supports them.
- Source speech is *read* speech (VCTK), not spontaneous speech; SOS-2SP matches conversational *timing* statistics to a real corpus but the linguistic content itself is still read material, not natural dialogue.
- Internal acoustic levels (LAeq) are relative/internal, not calibrated to real-world SPL (see "A note on levels" above).
- VCTK speaker pairing for SOS-2SP is fixed (not randomized per release); the same two speakers always appear together.

## License

This dataset is released under **[TODO: CC BY 4.0]**, consistent with the license of its dominant redistributed source material (VCTK Corpus, CC BY 4.0). The DEMAND background recordings and the Freesound event clips are used under their respective source licenses — see the links in "Source materials" above for the original terms. No CallHome/TalkBank material is redistributed (see note above).

**[TODO: confirm final license choice and any per-clip Freesound license terms before publishing.]**

## How to cite

```bibtex
@dataset{tailleur_2026_22687159,
  author       = {Tailleur, Modan and Chasle Cauchy, Théo and Lagrange, Mathieu},
  title        = {{SOS: Simulated Office Speech -- SOS-1SP and SOS-2SP}},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.22687159},
  url          = {https://doi.org/10.5281/zenodo.22687159}
}
```

**[TODO: add affiliations/ORCID iDs above — Zenodo will also generate its own "Cite as" block automatically on the record page once authors are set.]**

## Acknowledgements / source citations

- VCTK Corpus 0.92: Yamagishi, J., Veaux, C., MacDonald, K. (2019). *CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92). University of Edinburgh, CSTR. https://datashare.ed.ac.uk/handle/10283/3443
- DEMAND: Thiemann, J., Ito, N., Vincent, E. (2013). *The Diverse Environments Multichannel Acoustic Noise Database (DEMAND)*. https://www.kaggle.com/datasets/chrisfilo/demand
- CallHome English: TalkBank CallHome English corpus, used as a reference distribution only (no redistribution). https://talkbank.org/ca/access/CallHome/eng.html
- Freesound event recordings: "Keyboard typing in office" by user *bonnyorbit* (https://freesound.org/people/bonnyorbit/sounds/399823/); "Office ambience" by user *chelly01* (https://freesound.org/people/chelly01/sounds/541117/).
- Room acoustics simulation via [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (Scheibler, Bezzam, Dokmanić, 2018).

## Contact

**[TODO: contact email / institution]**
