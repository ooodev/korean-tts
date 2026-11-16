"""``Bingsu/KSS_Dataset`` -> the Parquet shards ``tiny_voca.prepare`` reads.

KSS ships 44.1 kHz mono; the model's acoustic grid is 24 kHz mono, so every clip
is resampled once here and stored as WAV bytes inside the shard.  ``expanded_script``
is the transcript (numbers and units already written out as spoken Korean), and
``voice_id`` is the single speaker name the model config declares.

Rows are kept when they fit the model: 0.5-15 s of audio (<= 3000 acoustic
frames) whose phoneme sequence is shorter than ``max_tokens``.  Splits are
deterministic from ``seed=42``.

    .venv/Scripts/python scripts/prepare_kss.py --out data
"""

import argparse
import io
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from scipy.signal import resample_poly

RATE = 24000
HOP = 256
MIN_SECONDS, MAX_SECONDS = 0.5, 15.0
MAX_TOKENS, MAX_FRAMES = 512, 3000
SPEAKER = "KSS"
SPLITS = (("train", 0.95), ("validation", 0.025), ("test", 0.025))


def convert(row, index=None):
    """One KSS row -> one shard row (24 kHz WAV bytes, transcript, ids, lengths)."""
    audio = row["audio"]
    wave, rate = sf.read(io.BytesIO(audio["bytes"])) if audio.get("bytes") is not None else sf.read(audio["path"])
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != RATE:
        factor = math.gcd(rate, RATE)
        wave = resample_poly(wave, RATE // factor, rate // factor).astype(np.float32)
    seconds = len(wave) / RATE
    frames = math.ceil(len(wave) / HOP)
    buffer = io.BytesIO()
    sf.write(buffer, wave, RATE, format="WAV", subtype="PCM_16")
    identifier = (audio.get("path") or "").rsplit("/", 1)[-1].rsplit(".", 1)[0] or f"kss-{index:05d}"
    return {"audio": {"bytes": buffer.getvalue(), "path": None},
            "id": identifier, "text_id": identifier,
            "voice_id": SPEAKER, "text": row["expanded_script"],
            "duration_seconds": seconds, "sample_rate": RATE, "frames": frames,
            "source_license": "CC BY-NC-SA 4.0",
            "source_url": "https://huggingface.co/datasets/Bingsu/KSS_Dataset"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="Bingsu/KSS_Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only the first N clips (smoke test)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source = load_dataset(args.dataset, split="train").cast_column("audio", Audio(decode=False))
    if args.limit:
        source = source.select(range(args.limit))
    print(json.dumps({"loaded": len(source)}), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda pair: convert(pair[1], pair[0]), enumerate(source)))
    print(json.dumps({"converted": len(rows)}), flush=True)

    from tiny_voca import text
    unknown_tokens, long_text, kept = 0, 0, []
    for row in rows:
        if not MIN_SECONDS <= row["duration_seconds"] <= MAX_SECONDS or row["frames"] > MAX_FRAMES:
            continue
        result = text.phonemize(row["text"])
        row["tokens"], row["stress"] = list(result.ids), list(result.stress)
        row["token_count"] = len(result.ids)
        unknown_tokens += len(result.unknown)
        if len(result.ids) >= MAX_TOKENS:
            long_text += 1
            continue
        kept.append(row)
    print(json.dumps({"kept": len(kept), "unknown_phonemes": unknown_tokens, "over_long": long_text}), flush=True)

    order = list(range(len(kept)))
    random.Random(args.seed).shuffle(order)
    counts, taken = {}, 0
    for name, share in SPLITS[:-1]:
        counts[name] = min(max(1, int(round(len(kept) * share))), len(kept) - taken - (len(SPLITS) - 2))
        taken += counts[name]
    counts[SPLITS[-1][0]] = len(kept) - taken
    start = 0
    for name, _ in SPLITS:
        take = order[start:start + counts[name]]
        start += counts[name]
        shard = args.out / f"{name}-00000-of-00001.parquet"
        payload = [{**{key: row[key] for key in ("audio", "id", "text_id", "voice_id", "text",
                                                 "duration_seconds", "sample_rate", "frames", "token_count",
                                                 "source_license", "source_url")},
                    "split": name}
                   for row, index in sorted(zip([kept[i] for i in take], take), key=lambda pair: pair[1])]
        Dataset.from_list(payload).to_parquet(shard)
        hours = sum(row["duration_seconds"] for row in [kept[i] for i in take]) / 3600
        print(json.dumps({"split": name, "rows": len(payload), "hours": round(hours, 3),
                          "shard": str(shard)}), flush=True)


if __name__ == "__main__":
    main()
