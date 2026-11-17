"""Build a single-speaker gate manifest from a merged annotation run.

The KO gate must measure only KSS rows, and the EN gate only F2/M2 rows, but
the merged manifest's test split mixes both, and the single-run manifests
(``annotations_full`` / ``annotations_en``) fail data.py's strict single-front-end
check when the serving process runs under the other front end. This copies the
requested splits filtered to the given speakers, keeping the parent manifest's
declared front-end/annotation sets so data.py accepts the rows unchanged:

    python scripts/gate_manifest.py --annotations annotations_mixed \
        --out eval/v2/korean --speaker KSS --splits test
    python scripts/gate_manifest.py --annotations annotations_mixed \
        --out eval/v2/english --speaker F2 --speaker M2 --splits test
"""
import argparse
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speaker", action="append", required=True,
                        help="Speaker to keep (repeatable; e.g. F2 and M2 for the EN gate)")
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args()

    parent = json.loads((args.annotations / "annotations.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in args.splits:
        kept = []
        with (args.annotations / (split + ".jsonl")).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and json.loads(line)["speaker"] in args.speaker:
                    kept.append(line)
        (args.out / (split + ".jsonl")).write_text("".join(kept), encoding="utf-8")
        counts[split] = len(kept)

    meta = {**parent, "counts": counts, "complete": True,
            "note": f"speaker-filtered ({', '.join(args.speaker)}) gate manifest of {args.annotations}; "
                    f"rows are an unmodified subset of the parent (complete) run"}
    (args.out / "annotations.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.out), "speakers": args.speaker, "counts": counts}))


if __name__ == "__main__":
    main()
