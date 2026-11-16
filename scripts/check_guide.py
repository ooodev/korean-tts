"""Sanity-check guide.html: placeholders resolved, audio files present, Korean intact."""
import re
from pathlib import Path

html = Path("guide.html").read_text(encoding="utf-8")
print("size", len(html) // 1024, "KB")
print("unfilled placeholders:", set(re.findall(r"\{[a-z_]+\}", html)) or "none")
sources = re.findall(r'src="([^"]+)"', html)
missing = [src for src in sources if not Path(src).exists()]
print("audio tags:", len(sources), "| missing files:", missing or "none")
print("tables:", html.count("<table>"), "| h2 sections:", len(re.findall(r"<h2 ", html)))
print("korean ok:", "안녕하세요. 오늘 날씨가 정말 맑고 좋네요." in html,
      "| codemix ok:", "paɪθ" in html, "| CER ok:", "11.8 %" in html)
