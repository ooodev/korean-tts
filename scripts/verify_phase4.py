"""Phase 4 gate: verify the 24 kHz Parquet splits that tiny_voca.prepare will read."""
import sys

import pyarrow.parquet as pq

for shard in sys.argv[1:]:
    table = pq.read_table(shard)
    rows = table.to_pylist()
    print("===", shard, "rows", len(rows))
    print("columns:", table.schema.names)
    first = rows[0]
    audio = first.pop("audio")
    print("row0:", {k: v for k, v in first.items()})
    print("audio:", {k: (f"<{len(v)} bytes>" if k == "bytes" and v else v) for k, v in audio.items()})
    ids = [r["id"] for r in rows]
    print("unique ids:", len(set(ids)), "empty ids:", sum(1 for i in ids if not i))
    print("splits:", {r["split"] for r in rows}, "voices:", {r["voice_id"] for r in rows},
          "sample rates:", {r["sample_rate"] for r in rows})
    durations = [r["duration_seconds"] for r in rows]
    print("seconds: min %.2f max %.2f total %.2f h" % (min(durations), max(durations), sum(durations) / 3600))
    print("empty text:", sum(1 for r in rows if not (r["text"] or "").strip()),
          "nan text:", sum(1 for r in rows if r["text"] is None))
