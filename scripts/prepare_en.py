"""The n4ze3m/tiny-voca English corpus -> Parquet shards for ``tiny_voca.prepare``.

Reads the pinned dataset snapshot from the HF cache (no network), applies the V2
screen (V2_SPEC.md section 3): Whisper WER <= 10%, 0.5-15 s, < 512 tokens, and
zero <unk> under the ``own`` front end (dizione). Writes one shard per split in the
KSS-compatible schema, so ``tiny_voca.prepare`` runs on it unchanged with the
default config (speakers F2/M2, 24 kHz).

    .venv/Scripts/python scripts/prepare_en.py [--out data_en] [--limit N]

``--limit N`` keeps only the first N accepted rows per split (smoke tests).
"""
import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "own")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if "DIZIONE_DATA" not in os.environ:
    candidate = Path(__file__).resolve().parents[1] / "weights" / "phonemizer"
    if (candidate / "lexicon.tsv").exists():
        os.environ["DIZIONE_DATA"] = str(candidate)
    else:
        raise SystemExit("DIZIONE_DATA is not set and no built lexicon was found")

import pyarrow as pa
import pyarrow.parquet as pq

from tiny_voca import text

REPO = "datasets--n4ze3m--tiny-voca"
RATE, HOP = 24000, 256
MIN_SECONDS, MAX_SECONDS = 0.5, 15.0
MAX_TOKENS = 512
WER_LIMIT = 0.10
SPLITS = ("train", "validation", "test")


def snapshot_data_dir():
    base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub" / REPO
    try:
        revision = (base / "refs" / "main").read_text().strip()
    except OSError as error:
        raise SystemExit(f"the {REPO} dataset snapshot is not cached (refs/main missing): {error}")
    data = base / "snapshots" / revision / "data"
    if not data.is_dir():
        raise SystemExit(f"snapshot data directory missing: {data}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data_en"))
    parser.add_argument("--limit", type=int, help="Keep only the first N accepted rows per split (smoke)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = snapshot_data_dir()
    print(json.dumps({"snapshot": str(data_dir)}), flush=True)

    from tiny_voca import text
    counts = {split: Counter() for split in SPLITS}
    kept = {split: [] for split in SPLITS}
    seen_ids = {}
    seen_texts = {}  # text key -> split that first admitted it
    for split in SPLITS:
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise SystemExit(f"no shards for split {split!r} in {data_dir}")
        for shard in files:
            for row in pq.read_table(shard).to_pylist():
                reason = screen(row)
                if reason:
                    counts[split][reason] += 1
                    continue
                if row["id"] in seen_ids:
                    counts[split]["duplicate_id"] += 1
                    continue
                seen_ids[row["id"]] = split
                # Repeated takes inside one split are valid rows (both voices read
                # many sentences); only text shared across two splits breaks
                # annotate's leakage gate, so drop those.
                key = row["text"].strip()
                if key in seen_texts and seen_texts[key] != split:
                    counts[split]["shared_text"] += 1
                    continue
                seen_texts.setdefault(key, split)
                kept[split].append(row)
                if args.limit and len(kept[split]) >= args.limit:
                    break
            if args.limit and len(kept[split]) >= args.limit:
                break
        print(json.dumps({"split": split, **dict(counts[split]), "kept": len(kept[split])}), flush=True)
        if not kept[split]:
            raise SystemExit(f"no usable rows for split {split!r}; inspect the counters above")

    for split in SPLITS:
        payload = [to_shard_row(row, split) for row in kept[split]]
        shard = args.out / f"{split}-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(payload), shard)
        hours = sum(row["duration_seconds"] for row in kept[split]) / 3600
        print(json.dumps({"split": split, "rows": len(payload), "hours": round(hours, 3), "shard": str(shard)}), flush=True)


def screen(row):
    """Return the drop reason, or None when the row passes the V2 screen."""
    audio = row.get("audio") or {}
    if audio.get("bytes") is None:
        return "no_audio_bytes"
    if not row.get("text") or not row["text"].strip():
        return "empty_text"
    if row.get("sample_rate") != RATE:
        return "wrong_sample_rate"
    wer = row.get("normalized_wer")
    if wer is None or wer > WER_LIMIT:
        return "wer_too_high"
    duration = row.get("duration_seconds")
    if duration is None or not MIN_SECONDS <= duration <= MAX_SECONDS:
        return "duration_out_of_range"
    result = text.phonemize(row["text"])
    if result.unknown:
        return "unknown_phonemes"
    if len(result.ids) >= MAX_TOKENS:
        return "too_many_tokens"
    return None


def to_shard_row(row, split):
    result = text.phonemize(row["text"])
    return {"audio": {"bytes": row["audio"]["bytes"], "path": None},
            "id": row["id"], "text_id": row.get("text_id"),
            "voice_id": row["voice_id"], "text": row["text"],
            "duration_seconds": row["duration_seconds"], "sample_rate": row["sample_rate"],
            "frames": math.ceil(row["duration_seconds"] * RATE / HOP),
            "token_count": len(result.ids),
            "source_license": row.get("source_license"), "source_url": row.get("source_url"),
            "origin": row.get("origin"), "source_id": row.get("source_id"),
            "teacher_model": row.get("teacher_model"), "reference_audio": row.get("reference_audio"),
            "split": split}


if __name__ == "__main__":
    main()
