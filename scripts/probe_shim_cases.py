"""Side-by-side g2pK analysis for a few sentences: real MeCab vs the shim."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g2pk import G2p
from mecab_shim import MeCab

SENTENCES = [
    "코치님, 저 잠시 쉬어야 할 것 같아요.",
    "제 아내 같은 여자는 세상에 둘도 없을 거예요.",
    "영화가 어찌나 지루한지 내내 잤어.",
    "그렇게 순수한 사람이 범죄를 저질렀다니 믿을 수가 없군.",
]

g2p = G2p()
out = []
real = g2p.mecab
for text in SENTENCES:
    g2p.mecab = real
    with_mecab = g2p(text)
    g2p.mecab = MeCab()
    with_shim = g2p(text)
    g2p.mecab = real
    out.append(text)
    out.append("  pos   mecab " + " ".join(f"{t}/{tag}" for t, tag in real.pos(text.split()[0] if False else text)))
    g2p.mecab = MeCab()
    out.append("  pos   shim  " + " ".join(f"{t}/{tag}" for t, tag in MeCab().pos(text)))
    g2p.mecab = real
    out.append("  jamo  mecab " + with_mecab)
    g2p.mecab = MeCab()
    out.append("  jamo  shim  " + with_shim)
    g2p.mecab = real
    out.append("")

Path("logs_shim_cases.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote logs_shim_cases.txt")
