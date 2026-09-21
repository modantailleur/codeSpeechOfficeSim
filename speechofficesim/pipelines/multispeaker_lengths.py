"""Match VCTK per-utterance word-length distribution to the CallHome corpus.

Produces ./data/{callhome,vctk,matched}.npy and ./data/matchedgroups.npy, the
latter being consumed by the multispeaker-group step. Ported from the
original reformat_vctk_multispeaker_files.py script.
"""
import os
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pylangacq


def _apply_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 29,
        "axes.labelsize": 33,
        "xtick.labelsize": 29,
        "ytick.labelsize": 29,
        "legend.fontsize": 29,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


# =========================================================
# CLEANING
# =========================================================
def clean_callhome_tokens(tokens):
    out = []

    for t in tokens:
        w = getattr(t, "word", None)
        if not w:
            continue

        if w.startswith("&=") or "@" in w:
            continue
        if w in {"xxx", "yyy", "www"}:
            continue
        if w.startswith("+"):
            continue
        if all(c in string.punctuation for c in w):
            continue

        w = re.sub(r"[^a-zA-Z']", "", w)
        if w:
            out.append(w.lower())

    return out


def clean_text(text):
    out = []
    for w in text.split():
        if not w:
            continue
        if all(c in string.punctuation for c in w):
            continue
        w = re.sub(r"[^a-zA-Z']", "", w)
        if w:
            out.append(w.lower())
    return out


# =========================================================
# CALLHOME LOADER
# =========================================================
def load_callhome(callhome_dir):
    files = list(Path(callhome_dir).rglob("*.cha"))
    print("[CALLHOME] files:", len(files))

    if len(files) == 0:
        raise RuntimeError("No .cha files found. Fix path.")

    lengths = []

    for f in files:
        try:
            chat = pylangacq.read_chat(f)
        except Exception:
            continue

        prev = None
        buf = []

        for utt in chat.utterances():
            spk = utt.participant
            words = clean_callhome_tokens(utt.tokens)

            if not words:
                continue

            if spk == prev:
                buf.extend(words)
            else:
                if buf:
                    lengths.append(len(buf))
                buf = words
                prev = spk

        if buf:
            lengths.append(len(buf))

    return np.array(lengths)


# =========================================================
# VCTK LOADER (returns filenames too)
# =========================================================
def load_vctk(vctk_txt_root, vctk_flac_root):
    utterances = []

    speakers = sorted(
        d for d in os.listdir(vctk_txt_root)
        if (vctk_txt_root / d).is_dir()
    )

    for spk in speakers:
        spk_dir = vctk_txt_root / spk

        for file in os.listdir(spk_dir):
            if not file.endswith(".txt"):
                continue

            flac_name = f"{Path(file).stem}_mic1.flac"
            flac_path = vctk_flac_root / spk / flac_name
            if not flac_path.exists():
                continue

            path = spk_dir / file

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            words = clean_text(text)

            if len(words) > 0:
                utterances.append((words, len(words), file))

    return utterances


def get_speaker_id(fname):
    # p279_086.txt -> p279
    return fname.split("_")[0]


# =========================================================
# MATCHING (returns groups)
# =========================================================
def match_vctk_to_callhome(vctk_utts, target_lengths, n=None, tries=20, log_every=50):
    if n is None:
        n = len(vctk_utts)

    speaker_to_indices = defaultdict(list)

    for idx, (_, L, fname) in enumerate(vctk_utts):
        spk = fname.split("_")[0]
        speaker_to_indices[spk].append(idx)

    sampled_targets = np.random.choice(
        target_lengths,
        size=n,
        replace=True
    )

    used_global = np.zeros(len(vctk_utts), dtype=bool)

    pseudo_lengths = []
    groups_all = []

    stop_all = False

    for i, target in enumerate(sampled_targets):

        if stop_all:
            break

        best_len = None
        best_used = None
        best_group = None
        best_valid = False

        for _ in range(tries):

            local_used = used_global.copy()

            available_speakers = [
                spk for spk, idxs in speaker_to_indices.items()
                if any(not local_used[j] for j in idxs)
            ]

            if not available_speakers:
                stop_all = True
                break

            speaker = random.choice(available_speakers)
            candidate_indices = speaker_to_indices[speaker][:]
            random.shuffle(candidate_indices)

            current_len = 0
            current_group = []

            for idx in candidate_indices:

                if local_used[idx]:
                    continue

                _, L, fname = vctk_utts[idx]

                # ---------------------------------------------
                # BAND-AWARE CONDITION
                # ---------------------------------------------
                if current_len + L <= 1.2 * target:

                    current_len += L
                    local_used[idx] = True
                    current_group.append(fname)

                    # stop early if already in range
                    if current_len >= 0.8 * target:
                        break

            # ---------------------------------------------
            # validate segment
            # ---------------------------------------------
            valid = (0.8 * target <= current_len <= 1.2 * target)

            # fallback if empty
            if len(current_group) == 0:
                for idx in candidate_indices:
                    if not local_used[idx]:
                        _, L, fname = vctk_utts[idx]
                        current_group = [fname]
                        current_len = L
                        local_used[idx] = True
                        break
                valid = (0.8 * target <= current_len <= 1.2 * target)

            # ---------------------------------------------
            # keep best VALID solution first
            # ---------------------------------------------
            if best_len is None:
                best_len = current_len
                best_used = local_used
                best_group = current_group
                best_valid = valid

            else:
                # prioritize valid solutions
                if valid and not best_valid:
                    best_len = current_len
                    best_used = local_used
                    best_group = current_group
                    best_valid = True

                elif valid == best_valid:
                    if abs(target - current_len) < abs(target - best_len):
                        best_len = current_len
                        best_used = local_used
                        best_group = current_group
                        best_valid = valid

        if best_used is None:
            break

        used_global = best_used

        pseudo_lengths.append(best_len)
        groups_all.append(best_group)

        if (i + 1) % log_every == 0 or (i + 1) == len(sampled_targets):
            current = np.array(pseudo_lengths)
            print(
                f"[MATCHING] {i+1}/{len(sampled_targets)} "
                f"({100*(i+1)/len(sampled_targets):.1f}%) | "
                f"mean={np.mean(current):.2f} | "
                f"std={np.std(current):.2f}"
            )

    # -------------------------------------------------
    # add remaining files
    # -------------------------------------------------
    used_files = set()

    for g in groups_all:
        used_files.update(g)

    for _, L, fname in vctk_utts:
        if fname not in used_files:
            groups_all.append([fname])
            pseudo_lengths.append(L)

    return np.array(pseudo_lengths), groups_all


# =========================================================
# PLOT (exact counts instead of histogram bins)
# =========================================================
def plot(callhome, vctk, figname="matched_distribution.png",
         titlename="Distribution Matching (Stochastic Bin Packing)", ymax=None):
    plt = _apply_plot_style()
    xlim = 40

    callhome = callhome[callhome <= xlim]
    vctk = vctk[vctk <= xlim]

    callhome_counts = Counter(callhome)
    vctk_counts = Counter(vctk)

    all_x = sorted(set(callhome_counts.keys()) | set(vctk_counts.keys()))

    callhome_y = [callhome_counts.get(x, 0) for x in all_x]
    vctk_y = [vctk_counts.get(x, 0) for x in all_x]

    plt.figure(figsize=(12, 5))

    plt.bar(all_x, callhome_y, alpha=0.6, label="CallHome")
    plt.bar(all_x, vctk_y, alpha=0.6, label="VCTK (matched to CallHome)")

    plt.xlabel("Words per speaker turn")
    plt.ylabel("Count")
    if ymax is not None:
        plt.ylim(top=ymax)

    plt.legend()

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.savefig(Path(figname).with_suffix(".pdf"))
    plt.show()


def plot_three(callhome, matched, vctk_lengths,
               figname="comparison_distribution.png",
               titlename="Word Count Distribution Comparison",
               ymax=None):
    plt = _apply_plot_style()
    xlim = 40

    callhome = callhome[callhome <= xlim]
    matched = matched[matched <= xlim]
    vctk_lengths = vctk_lengths[vctk_lengths <= xlim]

    callhome_counts = Counter(callhome)
    matched_counts = Counter(matched)
    vctk_counts = Counter(vctk_lengths)

    all_x = sorted(
        set(callhome_counts.keys())
        | set(matched_counts.keys())
        | set(vctk_counts.keys())
    )

    callhome_y = [callhome_counts.get(x, 0) for x in all_x]
    matched_y = [matched_counts.get(x, 0) for x in all_x]
    vctk_y = [vctk_counts.get(x, 0) for x in all_x]

    plt.figure(figsize=(12, 5))

    plt.plot(all_x, callhome_y, "-o", linewidth=2, markersize=4, color="tab:green",
             label="CallHome")
    plt.plot(all_x, vctk_y, "-^", linewidth=2, markersize=4, color="tab:orange",
             label="VCTK (original)")
    plt.plot(all_x, matched_y, "-s", linewidth=2, markersize=4, color="tab:blue",
             label="VCTK (matched to CallHome)")

    plt.xlabel("Words per speaker turn")
    plt.ylabel("Count")

    plt.margins(x=0, y=0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    if ymax is not None:
        plt.ylim(top=ymax)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.savefig(Path(figname).with_suffix(".pdf"))
    plt.show()


def plot_three_bis(callhome, matched, vctk_lengths,
                    figname="comparison_distribution.png",
                    titlename="Word Count Distribution Comparison",
                    ymax=None):
    plt = _apply_plot_style()
    xlim = 40

    callhome = callhome[callhome <= xlim]
    matched = matched[matched <= xlim]
    vctk_lengths = vctk_lengths[vctk_lengths <= xlim]

    callhome_counts = Counter(callhome)
    matched_counts = Counter(matched)
    vctk_counts = Counter(vctk_lengths)

    all_x = np.array(sorted(
        set(callhome_counts.keys())
        | set(matched_counts.keys())
        | set(vctk_counts.keys())
    ))

    callhome_y = [callhome_counts.get(x, 0) for x in all_x]
    matched_y = [matched_counts.get(x, 0) for x in all_x]
    vctk_y = [vctk_counts.get(x, 0) for x in all_x]

    plt.figure(figsize=(12, 5))

    plt.bar(
        all_x, callhome_y,
        color="tab:green",
        width=0.70,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.5,
        label="CallHome",
        zorder=1,
    )

    plt.bar(
        all_x, vctk_y,
        color="tab:orange",
        width=0.70,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.5,
        label="VCTK (original)",
        zorder=3,
    )

    plt.bar(
        all_x, matched_y,
        color="tab:blue",
        width=0.70,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.5,
        label="VCTK (matched to CallHome)",
        zorder=2,
    )

    plt.xlabel("Words per speaker turn")
    plt.ylabel("Count")

    if ymax is not None:
        plt.ylim(top=ymax)

    plt.xticks(all_x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.savefig(Path(figname).with_suffix(".pdf"))
    plt.show()


# =========================================================
# STATS
# =========================================================
def stats(name, arr):
    print(f"\n=== {name} ===")

    if len(arr) == 0:
        print("EMPTY")
        return

    print("N:", len(arr))
    print("Mean:", np.mean(arr))
    print("Std:", np.std(arr))
    print("Min:", np.min(arr))
    print("Max:", np.max(arr))


def run(callhome_dir="./dataset_CABank English CallHome Corpus",
        vctk_dir="../VCTK-Corpus-0.92",
        data_dir="./data"):
    random.seed(0)
    np.random.seed(0)

    vctk_txt_root = Path(vctk_dir) / "txt"
    vctk_flac_root = Path(vctk_dir) / "wav48_silence_trimmed"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    callhome_path = data_dir / "callhome.npy"

    print("\nLoading CallHome...")
    if callhome_path.exists():
        callhome = np.load(callhome_path)
        stats("CallHome", callhome)
        print("Loaded CallHome from callhome.npy")
    else:
        callhome = load_callhome(callhome_dir)
        stats("CallHome", callhome)
        np.save(callhome_path, callhome)
        print("Saved CallHome to callhome.npy")

    vctk_path = data_dir / "vctk.npy"
    vctk = None
    if vctk_path.exists():
        vctk_lengths = np.load(vctk_path)
        stats("VCTK", vctk_lengths)
        print("Loaded VCTK lengths from vctk.npy")
    else:
        print("\nLoading VCTK...")
        vctk = load_vctk(vctk_txt_root, vctk_flac_root)
        vctk_lengths = np.array([l for _, l, _ in vctk])
        stats("VCTK", vctk_lengths)
        np.save(vctk_path, vctk_lengths)
        print("Saved VCTK lengths to vctk.npy")

    matched_path = data_dir / "matched.npy"
    if matched_path.exists():
        matched = np.load(matched_path)
        stats("Matched VCTK", matched)
        print("Loaded matched lengths from matched.npy")
    else:
        print("\nMatching distributions...")
        if vctk is None:
            vctk = load_vctk(vctk_txt_root, vctk_flac_root)
        n = len(vctk)
        matched, groups = match_vctk_to_callhome(vctk, callhome, n=n)
        stats("Matched VCTK", matched)
        np.save(matched_path, matched)
        np.save(data_dir / "matchedgroups.npy", np.array(groups, dtype=object), allow_pickle=True)
        print("Saved matched lengths to matched.npy")

    print("\nPlotting...")
    plot_three(callhome, matched, vctk_lengths, figname="comparison_distribution.png",
               titlename="Word Count Distribution Comparison", ymax=8000)
