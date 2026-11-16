"""Re-split the KSS shards by sentence, so no transcript appears in two splits.

KSS records some sentences more than once, and a row-wise random split puts those
takes in different splits, which ``tiny_voca.annotate`` rejects as leakage. Every
recording of one transcript now goes to the same split; the target shares are the
same 95 / 2.5 / 2.5 as before, with the same seed.

    python scripts/resplit_kss.py [--data data] [--seed 42]
"""
import argparse
import random
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
SHARES = {"validation": 0.025, "test": 0.025}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    rows = []
    for split in SPLITS:
        rows.extend(pq.read_table(f"{arguments.data}/{split}-00000-of-00001.parquet").to_pylist())
    total = len(rows)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["text"].strip()].append(index)
    sentences = sorted(groups)
    random.Random(arguments.seed).shuffle(sentences)

    counts = Counter()
    assignment = {}
    for sentence in sentences:
        split = next((name for name in ("validation", "test") if counts[name] < SHARES[name] * total), "train")
        counts[split] += len(groups[sentence])
        for index in groups[sentence]:
            assignment[index] = split

    for split in SPLITS:
        table = pa.Table.from_pylist([dict(rows[index], split=split)
                                      for index in range(total) if assignment[index] == split])
        pq.write_table(table, f"{arguments.data}/{split}-00000-of-00001.parquet")
        seconds = sum(row["duration_seconds"] for row in table.to_pylist())
        print(f"{split:11s} rows {table.num_rows:6d}  hours {seconds / 3600:6.3f}  "
              f"sentences {len({row['text'].strip() for row in table.to_pylist()})}")
    print(f"total {total} rows, {len(sentences)} distinct sentences, seed {arguments.seed}")


if __name__ == "__main__":
    main()
