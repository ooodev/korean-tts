"""Merge per-part acoustic features back into one ``tiny_voca.annotate`` input directory.

``tiny_voca.prepare`` is single-process (``librosa.pyin`` dominates), so a shard is
prepared in parts by parallel processes and stitched together here:

    python scripts/merge_features.py features2 train features_w/train-0 features_w/train-1 ...

The journal key becomes ``<part>:<index>`` so it stays unique across the parts; the
feature ``.pt`` files are keyed by a hash of the clip id and never collide.
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    target, split, *sources = sys.argv[1:]
    target = Path(target)
    (target / split).mkdir(parents=True, exist_ok=True)

    rows, copied = 0, 0
    with (target / (split + ".jsonl")).open("w", encoding="utf-8") as out:
        for part, source in enumerate(sources):
            source = Path(source)
            for line in (source / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["record_key"] = f"{part}:{row['record_key'].split(':', 1)[1]}"
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
            for path in sorted((source / split).glob("*.pt")):
                shutil.copy2(path, target / split / path.name)
                copied += 1

    metadata = json.loads((Path(sources[0]) / (split + ".preparation.json")).read_text())
    metadata["complete"] = True
    metadata["accepted"] = rows
    metadata["merged_parts"] = [str(Path(source).resolve()) for source in sources]
    (target / (split + ".preparation.json")).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not (target / "features.json").exists():
        shutil.copy2(Path(sources[0]) / "features.json", target / "features.json")
    print(json.dumps({"split": split, "rows": rows, "features": copied,
                      "parts": len(sources), "manifest": str(target / (split + ".jsonl"))}))
    if copied != rows:
        raise SystemExit("a part is missing feature files for rows in its manifest")


if __name__ == "__main__":
    main()
