"""Slice the first N rows of a shard into a small smoke shard.

    python scripts/slice_shard.py data/train-00000-of-00001.parquet data_smoke/train-00000-of-00001.parquet 20
"""
import sys

import pyarrow.parquet as pq

source, target, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
table = pq.read_table(source).slice(0, count)
pq.write_table(table, target)
print(f"{source} -> {target}: {table.num_rows} rows")
