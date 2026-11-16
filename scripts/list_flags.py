"""Collect the exact CLI flags of every tiny_voca entry point (for WORKLOG.md)."""
import pathlib
import re

for name in ("prepare", "annotate", "train", "infer", "evaluate", "refine", "export"):
    source = pathlib.Path(f"tiny-voca/src/tiny_voca/{name}.py").read_text(encoding="utf-8")
    print("===", name, "===")
    for match in re.finditer(r"add_argument\(\s*(\"--[^\"]+\")(.*?)\)\n", source, re.S):
        flags = " ".join(match.group(2).split())
        print(f"  {match.group(1)} {flags[:150]}")
