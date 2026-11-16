"""Split a shard into N parts so several ``tiny_voca.prepare`` processes can run at once.

    python scripts/split_shards.py data/train-00000-of-00001.parquet data_parts/train-{index}.parquet 4
"""
import sys

import pyarrow.parquet as pq

source, target, parts = sys.argv[1], sys.argv[2], int(sys.argv[3])
table = pq.read_table(source)
size = -(-table.num_rows // parts)                      # ceil
for index in range(parts):
    start = index * size
    part = table.slice(start, min(size, table.num_rows - start))
    path = target.format(index=index)
    pq.write_table(part, path)
    print(f"{path}: {part.num_rows} rows")
