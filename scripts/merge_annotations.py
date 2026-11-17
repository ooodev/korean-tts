"""Merge the KSS (ko) and English (own) annotation runs into one bilingual manifest.

Training reads a single acoustic root; to make that work without copying gigabytes,
``<features-mixed>/ko`` and ``<features-mixed>/en`` are Windows directory junctions
onto the two feature directories, and the merged rows carry ``ko/``/``en/`` path
prefixes. The merged ``annotations.json`` declares the front-end and annotation
fingerprints of both runs (``data.py`` accepts a row when its values are in the
declared sets), and every sub-run's full identity is embedded for provenance.

    .venv/Scripts/python scripts/merge_annotations.py \
        --ko annotations_full --en annotations_en --out annotations_mixed \
        --features-mixed features_mixed --features-ko features2 --features-en features_en
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SPLITS = ("train", "validation", "test")


def make_junction(link, target):
    if link.exists():
        if link.resolve() == target:
            return
        # Repoint: remove the junction itself (never its contents), then recreate.
        subprocess.run(["cmd", "/c", "rmdir", str(link)], check=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)
    print(f"junction {link} -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ko", type=Path, required=True, help="KSS annotation directory")
    parser.add_argument("--en", type=Path, required=True, help="English annotation directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features-mixed", type=Path, default=Path("features_mixed"))
    parser.add_argument("--features-ko", type=Path, default=Path("features2"))
    parser.add_argument("--features-en", type=Path, default=Path("features_en"))
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Accept a --limit smoke run (its manifest says complete: false)")
    args = parser.parse_args()

    sources = {"ko": args.ko, "en": args.en}
    metadata = {}
    for name, source in sources.items():
        meta = json.loads((source / "annotations.json").read_text(encoding="utf-8"))
        if not meta.get("complete") and not args.allow_incomplete:
            raise SystemExit(f"{source} is not complete; finish it with --resume first "
                             f"(or pass --allow-incomplete for a deliberate smoke merge)")
        metadata[name] = meta
    ko_fp, en_fp = metadata["ko"]["fingerprint"], metadata["en"]["fingerprint"]
    if ko_fp == en_fp:
        raise SystemExit("the two annotation runs have the same fingerprint; nothing to merge")

    make_junction(args.features_mixed / "ko", args.features_ko.resolve())
    make_junction(args.features_mixed / "en", args.features_en.resolve())

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        rows, rejected = [], []
        for name, prefix in (("ko", "ko/"), ("en", "en/")):
            with (sources[name] / (split + ".jsonl")).open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        row["path"] = prefix + row["path"]
                        rows.append(json.dumps(row, ensure_ascii=False))
            rejected_path = sources[name] / (split + ".rejected.jsonl")
            if rejected_path.exists():
                rejected.append((name, rejected_path))
        target = args.out / (split + ".jsonl")
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        temporary.replace(target)
        rejected_count = 0
        for name, path in rejected:
            shutil.copy2(path, args.out / f"{name}-{split}.rejected.jsonl")
            rejected_count += sum(1 for line in path.open(encoding="utf-8") if line.strip())
        counts[split] = {"accepted": len(rows), "rejected": rejected_count,
                         "ko": metadata["ko"]["counts"][split], "en": metadata["en"]["counts"][split]}

    merged_fp = hashlib.sha256(json.dumps({"ko": ko_fp, "en": en_fp}, sort_keys=True).encode()).hexdigest()
    metadata_out = {
        "annotation_version": 3,
        "merged": True,
        "fingerprint": merged_fp,
        # Declared sets accepted by data.py: one front end and one annotation run per corpus.
        "frontends": {metadata["ko"]["identity"]["frontend"]: "ko (g2pK + kresnik aligner)",
                      metadata["en"]["identity"]["frontend"]: "own (dizione + espeak-CTC aligner)"},
        "annotations": {ko_fp: "annotations_full (KSS)", en_fp: "annotations_en (tiny-voca EN)"},
        "acoustic_root": str(args.features_mixed),
        "complete": bool(metadata["ko"].get("complete") and metadata["en"].get("complete")),
        "counts": {split: counts[split]["accepted"] for split in SPLITS},
        "details": counts,
        "sources": {"ko": ko_fp, "en": en_fp},
        "identities": {"ko": metadata["ko"]["identity"], "en": metadata["en"]["identity"]},
    }
    (args.out / "annotations.json").write_text(json.dumps(metadata_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"merged": str(args.out), "fingerprint": merged_fp[:16], "counts": metadata_out["counts"]}))


if __name__ == "__main__":
    main()
