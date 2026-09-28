"""Command-line entry point: `python -m speechofficesim <subcommand> [flags]`.

Each subcommand corresponds to one step of the monospeaker or multispeaker
dataset-building pipeline described in the README. Subcommand modules are
imported lazily so that e.g. `--help` and unrelated subcommands stay fast and
don't pull in matplotlib/pyroomacoustics/pylangacq.
"""
import argparse
import sys


def _cmd_monospeaker_reformat(args):
    from .pipelines import monospeaker
    monospeaker.run(input_vctk_dir=args.input_vctk_dir, output_vctk_dir=args.output_vctk_dir)


def _cmd_multispeaker_lengths(args):
    from .pipelines import multispeaker_lengths
    multispeaker_lengths.run(callhome_dir=args.callhome_dir, vctk_dir=args.vctk_dir, data_dir=args.data_dir)


def _cmd_multispeaker_group(args):
    from .pipelines import multispeaker_group
    multispeaker_group.run(groups_path=args.groups_path, input_vctk_dir=args.input_vctk_dir, output_vctk_dir=args.output_vctk_dir)


def _cmd_multispeaker_conversations(args):
    from .pipelines import multispeaker_conversations
    multispeaker_conversations.run(input_vctk_dir=args.input_vctk_dir, output_vctk_dir=args.output_vctk_dir)


def _cmd_mix(args):
    from .pipelines import mixing
    mixing.run(
        sr=args.sr,
        events_path=args.events_path,
        background_path=args.background_path,
        segment_duration=args.segment_duration,
        vctk_path=args.vctk_path,
        out_dataset_path=args.out_dataset_path,
    )


def _cmd_vad(args):
    from .pipelines import vad
    vad.run(dataset_path=args.dataset_path)


def _cmd_office_level(args):
    from .pipelines import office_level
    office_level.run(
        keyboard_path=args.keyboard_path,
        office_path=args.office_path,
        sr=args.sr,
        output_path=args.output_path,
    )


def _cmd_check_integrity(args):
    from .pipelines import integrity_check
    _n_checked, n_failed = integrity_check.run(
        dataset_path=args.dataset_path,
        num_speakers=args.num_speakers,
        num_segments=args.num_segments,
        events_path=args.events_path,
        background_path=args.background_path,
        check_ebr=args.check_ebr,
    )
    sys.exit(1 if n_failed else 0)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="speechofficesim",
        description="Build simulated speech-in-office datasets from VCTK, DEMAND and CallHome.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- monospeaker pipeline ---
    p = subparsers.add_parser("monospeaker-reformat", help="Group VCTK utterances into ~1min monospeaker files.")
    p.add_argument("--input_vctk_dir", type=str, default="../VCTK-Corpus-0.92")
    p.add_argument("--output_vctk_dir", type=str, default="../VCTK-Corpus-0.92-MONOSPEAKER")
    p.set_defaults(func=_cmd_monospeaker_reformat)

    # --- multispeaker pipeline ---
    p = subparsers.add_parser("multispeaker-lengths", help="Match VCTK word-length distribution to CallHome and save speaker groups.")
    p.add_argument("--callhome_dir", type=str, default="./dataset_CABank English CallHome Corpus")
    p.add_argument("--vctk_dir", type=str, default="../VCTK-Corpus-0.92")
    p.add_argument("--data_dir", type=str, default="./data")
    p.set_defaults(func=_cmd_multispeaker_lengths)

    p = subparsers.add_parser("multispeaker-group", help="Copy VCTK and concatenate utterances into the matched groups.")
    p.add_argument("--groups_path", type=str, default="./data/matchedgroups.npy")
    p.add_argument("--input_vctk_dir", type=str, default="../VCTK-Corpus-0.92")
    p.add_argument("--output_vctk_dir", type=str, default="../VCTK-Corpus-0.92-GROUPED")
    p.set_defaults(func=_cmd_multispeaker_group)

    p = subparsers.add_parser("multispeaker-conversations", help="Build ~1min two-speaker conversations from the grouped VCTK corpus.")
    p.add_argument("--input_vctk_dir", type=str, default="../VCTK-Corpus-0.92-GROUPED")
    p.add_argument("--output_vctk_dir", type=str, default="../VCTK-Corpus-0.92-CONVERSATIONS")
    p.set_defaults(func=_cmd_multispeaker_conversations)

    # --- shared mixing + VAD steps ---
    p = subparsers.add_parser("mix", help="Mix VCTK speech with office events and background noise.")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--events_path", type=str, default="./audio/office_events.wav")
    p.add_argument("--background_path", type=str, default="./audio/ch01ch04-ooffice-demand.wav")
    p.add_argument("--segment_duration", type=int, default=60)
    p.add_argument("--vctk_path", type=str, default="../VCTK-Corpus-0.92-MONOSPEAKER")
    p.add_argument("--out_dataset_path", type=str, default="../SOS-1SP/")
    p.set_defaults(func=_cmd_mix)

    p = subparsers.add_parser("vad", help="Compute ground-truth speech-activity annotations for a mixed dataset.")
    p.add_argument("--dataset_path", type=str, default="../SOS-1SP",
                    help="Path to the dataset root directory.")
    p.set_defaults(func=_cmd_vad)

    # --- diagnostics ---
    p = subparsers.add_parser("office-level", help="Report the combined LAeq of the keyboard + office ambience source clips.")
    p.add_argument("--keyboard_path", type=str, default="./audio/399823__bonnyorbit__keyboard-typing-in-office.wav")
    p.add_argument("--office_path", type=str, default="./audio/541117__chelly01__office-ambience.wav")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--output_path", type=str, default="combined_audio.wav")
    p.set_defaults(func=_cmd_office_level)

    p = subparsers.add_parser(
        "check-integrity",
        help="Audit a generated dataset's EBR/SAR levels against the mixing pipeline's own definitions (no VCTK needed).",
    )
    p.add_argument("--dataset_path", type=str, required=True, help="Root of a generated SOS-1SP/SOS-2SP dataset.")
    p.add_argument("--num_speakers", type=int, default=1, help="How many speaker/pair ids to check (first N, sorted).")
    p.add_argument("--num_segments", type=int, default=1, help="How many segments per speaker/pair to check (first M, sorted).")
    p.add_argument("--events_path", type=str, default="./audio/office_events.wav")
    p.add_argument("--background_path", type=str, default="./audio/ch01ch04-ooffice-demand.wav")
    p.add_argument("--check_ebr", action="store_true",
                    help="Also run the EBR check (best-effort, ~30s per file via FFT cross-correlation; SAR check alone is instant).")
    p.set_defaults(func=_cmd_check_integrity)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
