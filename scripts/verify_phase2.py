"""Phase 2 exit gate: dizione English phonemization + g2pk Korean G2P both run."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DIZIONE_DATA", str(ROOT / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import dizione  # noqa: E402
import dizione.engine  # noqa: E402,F401  (populates the dizione.engine submodule)
from dizione.symbols import SYMBOLS  # noqa: E402
from g2pk import G2p  # noqa: E402

ok = True

print(f"dizione {dizione.__version__} (VERSION={dizione.VERSION}) data={dizione.engine.default_directory()}")
print(f"dizione len(SYMBOLS) = {len(SYMBOLS)}")
res = dizione.phonemize("Hello, world!")
print("dizione phonemize('Hello, world!') ->", str(res))
print("  ids    =", res.ids)
print("  stress =", res.stress)
print("  unknown=", res.unknown)
assert res.symbols[0] == "<bos>" and res.symbols[-1] == "<eos>", "dizione must wrap with BOS/EOS"
assert len(res.ids) == len(res.stress) == len(res.symbols)
assert any(s > 0 for s in res.stress[1:-1]), "expected non-zero lexical stress"

res2 = dizione.phonemize("I read the book you read yesterday.")
print("dizione phonemize(homograph) ->", str(res2))

ident = dizione.identity()
print("dizione identity fingerprint =", ident["fingerprint"][:16], "built files:", sorted(ident["built"]))

g2p = G2p()
out = g2p("맑은 날씨가 좋아요.")
print("g2pk('맑은 날씨가 좋아요.') ->", out)
expected = "말근 날씨가 조아요."
if out != expected:
    print(f"MISMATCH: expected {expected!r}")
    ok = False
out2 = g2p("안녕하세요. 국물 맛이 끝내주고 신라의 역사도 흥미롭습니다.")
print("g2pk(long) ->", out2)

import torch  # noqa: E402
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

import tiny_voca  # noqa: E402
from tiny_voca.config import Config  # noqa: E402
from tiny_voca import text as tvtext  # noqa: E402
print("tiny_voca frontend:", tvtext.frontend(), "len(SYMBOLS):", len(tvtext.SYMBOLS))
print("Config():", Config().to_dict()["vocab_size"], Config().speakers)
ph = tvtext.phonemize("Hello, world!")
print("tiny_voca.phonemize ->", " ".join(ph.symbols))
assert ph.ids == res.ids and ph.stress == res.stress, "tiny_voca 'own' frontend must equal dizione output"

import pyarrow, librosa, soundfile, transformers, torchaudio, jiwer, faster_whisper  # noqa: E402,F401
print("pyarrow", pyarrow.__version__, "| librosa", librosa.__version__,
      "| transformers", transformers.__version__, "| torchaudio", torchaudio.__version__)

print("PHASE2 OK" if ok else "PHASE2 FAIL")
sys.exit(0 if ok else 1)
