"""Build the browser front end's morphology dictionary from the corpus.

    python scripts/build_mecab_dict.py --train annotations_full/train.jsonl \
        --output tiny-voca/web/public/assets/mecab-dict.json

``scripts/mecab_shim.py`` is a pure-Python stand-in for ``python-mecab-ko``, which
Pyodide cannot run. Its tables get the analysis right most of the time; this
dictionary makes it exact where the corpus has seen the word. It maps an eojeol
to the morpheme/tag sequence the real MeCab produced for it, keeping the most
frequent analysis per eojeol, so the browser reproduces the front end the model
was trained on for the words the corpus contains.

The dictionary is built from the training split only, so validation and test
measure real coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")

from g2pk import G2p

from g2pk import G2p

_TOKEN = re.compile(r"[^\s]+")


def eojeols(text: str) -> list[str]:
    return _TOKEN.findall(text)


def texts_from_parquet(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text", "split"])
    return [text for text, split in zip(table["text"].to_pylist(), table["split"].to_pylist())
            if split == "train" and text]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train-00000-of-00001.parquet"),
                        help="a KSS Parquet shard (all 12 210 train sentences) or a JSONL manifest")
    parser.add_argument("--output", type=Path,
                        default=Path("tiny-voca/web/public/assets/mecab-dict.json"))
    parser.add_argument("--min-count", type=int, default=1,
                        help="drop analyses seen fewer than this many times")
    args = parser.parse_args()

    g2p = G2p()
    counts: dict[str, Counter] = defaultdict(Counter)
    if args.train.suffix == ".parquet":
        sentences = texts_from_parquet(args.train)
    else:
        sentences = [json.loads(line)["text"] for line in args.train.open(encoding="utf-8")]
    for text in sentences:
        for eojeol in eojeols(text):
            # g2pK keeps only tag.split("+")[-1][0], so the dictionary stores
            # exactly that: (morpheme, first tag letter) pairs.
            analysis = tuple(
                (morph, tag.split("+")[-1][:1]) for morph, tag in g2p.mecab.pos(eojeol)
            )
            counts[eojeol][analysis] += 1

    # One analysis per eojeol: the most frequent one, ties broken by the shorter
    # (more general) segmentation.
    table = {}
    for eojeol, analyses in counts.items():
        best = max(analyses.items(), key=lambda item: (item[1], -len(item[0])))[0]
        total = sum(analyses.values())
        if total >= args.min_count:
            table[eojeol] = [[morph, tag] for morph, tag in best]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    tokens = sum(sum(c.values()) for c in counts.values())
    covered = sum(sum(c.values()) for eojeol, c in counts.items() if eojeol in table)
    print(f"{len(sentences)} sentences, {tokens} eojeols, {len(counts)} distinct")
    print(f"dictionary: {len(table)} eojeols, {args.output.stat().st_size / 1e6:.2f} MB "
          f"-> {args.output}")
    print(f"eojeol coverage on the training split: {100 * covered / tokens:.2f} %")


if __name__ == "__main__":
    main()
