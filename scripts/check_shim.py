"""Gate: the pure-Python MeCab stand-in must reproduce the front end's tokens.

    python scripts/check_shim.py --sentences 3000

``scripts/mecab_shim.py`` replaces ``python-mecab-ko`` in the browser (Pyodide has
no native extensions). g2pK only reads the first letter of each POS tag, so the
question is how often the shim's segmentation changes the Jamo ids the trainer
saw. Reports the share of sentences whose token stream is identical, the mean
token edit distance, and the most common divergences.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")

from tiny_voca import korean
from tiny_voca.korean import phonemize_mixed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mecab_shim import MeCab  # noqa: E402


def read_sentences(paths: list[Path], limit: int, seed: int) -> list[str]:
    texts = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line).get("text")
                if text:
                    texts.append(text)
    random.Random(seed).shuffle(texts)
    return texts[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+",
                        default=[Path("annotations_full/validation.jsonl"),
                                 Path("annotations_full/test.jsonl")],
                        help="held-out splits: the dictionary is built from train only")
    parser.add_argument("--sentences", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", type=int, default=6)
    parser.add_argument("--diffs", type=int, default=0, help="show and count the first disagreeing spans")
    parser.add_argument("--report", default=None, help="write the UTF-8 diff report here")
    args = parser.parse_args()

    sentences = read_sentences(args.manifest, args.sentences, args.seed)
    analyser = korean.g2p()                  # one G2p, its .mecab is swappable
    reference = [phonemize_mixed(text) for text in sentences]
    reference_ids = [result.ids for result in reference]

    analyser.mecab = MeCab()                 # swap in the pure-Python stand-in
    candidate = [phonemize_mixed(text) for text in sentences]
    candidate_ids = [result.ids for result in candidate]
    known = sum(1 for text in sentences
                if all(word in analyser.mecab.dictionary or not any("가" <= c <= "힣" for c in word)
                       for word in re.findall(r"[^\s]+", text)))

    same = sum(a == b for a, b in zip(reference_ids, candidate_ids))
    matcher = [SequenceMatcher(None, a, b) for a, b in zip(reference_ids, candidate_ids)]
    distance = [sum(size for _, _, size in m.get_matching_blocks()) for m in matcher]
    tokens = [len(a) for a in reference_ids]
    edits = sum(t - kept for t, kept in zip(tokens, distance))
    total = sum(tokens)

    print(f"{len(sentences)} sentences, {total} tokens from " + ", ".join(p.name for p in args.manifest))
    print(f"every eojeol in the dictionary: {known / len(sentences):6.2%} ({known}/{len(sentences)})")
    print(f"identical token stream : {same / len(sentences):6.2%} ({same}/{len(sentences)})")
    print(f"token accuracy         : {100 * (total - edits) / total:6.2f} % ({edits} tokens differ)")
    print(f"mean tokens per sentence {total / len(sentences):.1f}, "
          f"mean edit {edits / len(sentences):.2f}")

    if args.diffs:
        # Where do the two analyses disagree? Print the first differing span with
        # context, and count the shape of the disagreement.
        shapes = Counter()
        shown = 0
        for index, (a, b) in enumerate(zip(reference_ids, candidate_ids)):
            if a == b:
                continue
            match = SequenceMatcher(None, a, b)
            for tag, i1, i2, j1, j2 in match.get_opcodes():
                if tag == "equal":
                    continue
                left = " ".join(reference[index].symbols[max(0, i1 - 2):i1])
                real = " ".join(reference[index].symbols[i1:i2])
                shim = " ".join(candidate[index].symbols[j1:j2])
                right = " ".join(reference[index].symbols[i2:i2 + 2])
                shapes[f"{tag}: [{shim}] -> [{real}]"] += 1
                if shown < args.diffs:
                    shown += 1
                    print(f"\n  {sentences[index]}")
                    print(f"    mecab …{left} ⟦{real}⟧ {right}")
                    print(f"    shim  …{left} ⟦{shim}⟧ {right}")
        print("\nmost frequent disagreements:")
        for shape, count in shapes.most_common(12):
            print(f"  {count:5d}  {shape}")
        if args.report:
            lines = [f"{same / len(sentences):.2%} identical, {sum(shapes.values())} disagreeing spans",
                     "", "most frequent disagreements:"]
            lines += [f"  {count:5d}  {shape}" for shape, count in shapes.most_common(25)]
            lines += ["", "examples:"]
            for index, (a, b) in enumerate(zip(reference_ids, candidate_ids)):
                if a == b:
                    continue
                match = SequenceMatcher(None, a, b)
                for tag, i1, i2, j1, j2 in match.get_opcodes():
                    if tag == "equal":
                        continue
                    lines += [f"  {sentences[index]}",
                              "    mecab …" + " ".join(reference[index].symbols[max(0, i1 - 3):i2 + 3]),
                              "    shim  …" + " ".join(candidate[index].symbols[max(0, j1 - 3):j2 + 3])]
                    break
            Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"report -> {args.report}")

    if args.show:
        worst = sorted(range(len(sentences)),
                       key=lambda i: SequenceMatcher(None, reference_ids[i], candidate_ids[i]).ratio())
        print("\nmost divergent:")
        for index in worst[: args.show]:
            print(f"  {sentences[index]}")
            print(f"    mecab {' '.join(reference[index].symbols)}")
            print(f"    shim  {' '.join(candidate[index].symbols)}")

    ok = same / len(sentences) >= 0.90
    print(f"\n{'OK' if ok else 'BELOW GATE'}: 90 % identical required")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
