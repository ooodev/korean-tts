"""A reproducible training subset of a (merged) annotation manifest, for smoke runs.

Stratified by speaker so every voice is present, and every selected feature file is
verified to exist before the subset is declared complete. The subset's
``annotations.json`` keeps the declared front-end/annotation sets of its parent and
carries its own fingerprint, so ``data.py`` and ``train.py`` treat it like any run.

    .venv/Scripts/python scripts/subset_manifest.py \
        --annotations annotations_mixed --out annotations_mixed_smoke \
        --n-train 60 --n-validation 20 --n-test 20 --seed 42
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SPLITS = ("train", "validation", "test")


def pick(rows, n, rng):
    """n rows stratified by speaker (at least one of each voice when available)."""
    if n >= len(rows):
        return rows
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker"], []).append(row)
    chosen, remaining = [], n
    for speaker in sorted(by_speaker):
        pool = by_speaker[speaker]
        take = min(1, len(pool))
        chosen.extend(pool[:take])
        remaining -= take
        if remaining <= 0:
            break
    rest = [row for speaker in sorted(by_speaker) for row in by_speaker[speaker][1:]]
    if remaining > 0:
        chosen.extend(rng.sample(rest, min(remaining, len(rest))))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-validation", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Accept a parent merged from --limit smoke runs")
    args = parser.parse_args()

    parent = json.loads((args.annotations / "annotations.json").read_text(encoding="utf-8"))
    if not parent.get("complete") and not args.allow_incomplete:
        raise SystemExit("parent manifest is not complete "
                         "(pass --allow-incomplete for a deliberate smoke subset)")
    root = Path(parent["acoustic_root"])
    limits = {"train": args.n_train, "validation": args.n_validation, "test": args.n_test}

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        rows = [json.loads(line) for line in
                (args.annotations / (split + ".jsonl")).read_text(encoding="utf-8").split("\n") if line.strip()]
        rng = random.Random(f"{args.seed}:{split}")
        subset = pick(rows, limits[split], rng)
        for row in subset:
            if not (root / row["path"]).is_file():
                raise SystemExit(f"missing feature file for {row['id']}: {root / row['path']}")
        temporary = (args.out / split).with_name(split + ".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset), encoding="utf-8")
        temporary.replace(args.out / (split + ".jsonl"))
        counts[split] = {"accepted": len(subset),
                         "by_speaker": {speaker: sum(r["speaker"] == speaker for r in subset)
                                        for speaker in sorted({r["speaker"] for r in subset})}}

    seed_fp = hashlib.sha256(json.dumps({"parent": parent["fingerprint"], "seed": args.seed,
                                         "n": limits}, sort_keys=True).encode()).hexdigest()
    metadata = {
        "annotation_version": parent["annotation_version"],
        "merged": parent.get("merged", False),
        "subset_of": parent["fingerprint"],
        "fingerprint": seed_fp,
        "acoustic_root": parent["acoustic_root"],
        "complete": True,
        "counts": {split: counts[split]["accepted"] for split in SPLITS},
        "details": counts,
    }
    # Declared sets: carry the parent's forward so merged corpora keep validating.
    if parent.get("frontends") is not None:
        metadata["frontends"] = parent["frontends"]
    if parent.get("annotations") is not None:
        metadata["annotations"] = parent["annotations"]
    (args.out / "annotations.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"subset": str(args.out), "fingerprint": seed_fp[:16], "counts": metadata["counts"]}))


if __name__ == "__main__":
    main()
