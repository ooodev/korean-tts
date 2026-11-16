"""Phase 5 gate, part 2: structural + acoustic sanity of CTC syllable alignments.

Reads an annotation manifest (``annotations/<split>.jsonl``) written by
``tiny_voca.annotate`` and asserts, for every clip:

* ``len(tokens) == len(stress) == len(durations)``
* ``sum(durations) == frames`` on the 24 kHz / hop-256 grid
* every duration >= 0 and every *spoken* token (English phone or Korean Jamo) >= 1 frame
* the alignment did not collapse: the longest phone is a sane share of the clip
* QC confidence (mean CTC posterior of the aligned syllables) is reasonable

Then prints three alignments as ``token | duration_ms`` for human audit.
"""

import json
import sys
from collections import Counter

import numpy as np

from tiny_voca.text import PHONE_SET, SYMBOLS

FRAME_MS = 256 / 24000 * 1000
MAX_SHARE = 0.35                       # no single phone may own more than a third of a clip


def main(manifest, show=3):
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("the manifest is empty")
    failures, durations_all, rates, probabilities = [], [], [], []
    for row in rows:
        tokens, stress, durations, frames = row["tokens"], row["stress"], row["durations"], row["frames"]
        problems = []
        if not (len(tokens) == len(stress) == len(durations)):
            problems.append("length mismatch")
        if sum(durations) != frames:
            problems.append(f"sum {sum(durations)} != frames {frames}")
        if min(durations) < 0:
            problems.append("negative duration")
        spoken = [d for token, d in zip(tokens, durations) if token in PHONE_ID_SET]
        if spoken and min(spoken) < 1:
            problems.append("a phone has no frame")
        if max(durations) > MAX_SHARE * frames:
            problems.append(f"collapsed: longest phone {max(durations)}/{frames} frames")
        if row["qc"]["token_probability_mean"] < 0.3:
            problems.append(f"low posterior {row['qc']['token_probability_mean']}")
        if problems:
            failures.append((row["id"], problems))
        durations_all.extend(spoken)
        rates.append(row["qc"]["phones_per_second"])
        probabilities.append(row["qc"]["token_probability_mean"])

    print(f"clips: {len(rows)}  failures: {len(failures)}")
    for identifier, problems in failures[:10]:
        print("  FAIL", identifier, problems)
    quantiles = np.percentile(durations_all, [0, 50, 95, 100])
    print("phone frames  p0/p50/p95/max: %s" % np.round(quantiles, 1).tolist())
    print("phones/second mean: %.2f   token posterior mean: %.3f" % (np.mean(rates), np.mean(probabilities)))
    print("qc quantiles (all clips):")
    for key in ("token_probability_mean", "weak_phone_fraction", "max_phone_frames", "vowel_voiced",
                "voiceless_voiced", "pause_level_db", "leading_frames", "trailing_frames"):
        values = [row["qc"][key] for row in rows if row["qc"].get(key) is not None]
        if values:
            print(f"  {key:24s} p5={np.percentile(values, 5):8.3f} p50={np.percentile(values, 50):8.3f} "
                  f"p95={np.percentile(values, 95):8.3f}")

    for row in rows[:show]:
        print("\n---", row["id"], f"{row['frames']} frames = {row['frames'] * FRAME_MS:.0f} ms")
        print(row["text"])
        cells = [f"{SYMBOLS[token]}|{duration * FRAME_MS:.0f}" for token, duration in zip(row["tokens"], row["durations"])]
        print("  token|ms  " + "  ".join(cells))
    return 1 if failures else 0


PHONE_ID_SET = frozenset(SYMBOLS.index(symbol) for symbol in PHONE_SET)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
