"""Markdown sanity check for the workspace docs: balanced fences, table shape, links."""
import re
import sys
from pathlib import Path

for name in sys.argv[1:] or ["AGENTS.md", "DATASET.md", "TRAIN.md", "GUIDE.md"]:
    path = Path(name)
    text = path.read_text(encoding="utf-8")
    fences = len(re.findall(r"^```", text, flags=re.M))
    rows = [line for line in text.splitlines() if line.startswith("|")]
    widths = {len(line.split("|")) for line in rows}
    headings = re.findall(r"^#{1,3} (.+)", text, flags=re.M)
    print(f"{name:12s} {len(text)/1024:5.1f} KB  fences {fences:3d} "
          f"({'ok' if fences % 2 == 0 else 'UNBALANCED'})  table rows {len(rows):4d}  "
          f"columns {sorted(widths) if widths else '-'}")
    print(f"             sections: {', '.join(h for h in headings if h.startswith(('1','2','3','4','5','6','7','8','9')) or ' ' in h)[:150]}")
