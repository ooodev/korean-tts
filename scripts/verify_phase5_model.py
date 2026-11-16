"""Phase 5 gate, part 1: the model instantiates and runs on the 142-token Korean inventory.

Loads config-ko.json, builds TrainingSystem, runs a forward pass with dummy tensors
shaped like a real KSS clip, and reports the parameter budget train.py enforces.
"""

import json
import os

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")
os.environ.setdefault("DIZIONE_DATA", r"C:\ai-workspace\korean-tts\weights\phonemizer")

import torch

from tiny_voca.config import Config
from tiny_voca.data import collate
from tiny_voca.train import TrainingSystem
from tiny_voca.text import PHONE_SET, SYMBOLS, phonemize

config = Config.from_dict(json.loads(open("config-ko.json", encoding="utf-8").read()))
print("frontend inventory:", len(SYMBOLS), "config vocab_size:", config.vocab_size)
assert config.vocab_size == len(SYMBOLS) == 142

system = TrainingSystem(config)
total = sum(p.numel() for p in system.model.parameters())
print("inference parameters:", f"{total:,}", "(train.py allows up to 5,000,000)")

text = "안녕하세요. 오늘 날씨가 정말 맑고 좋네요."
result = phonemize(text)
count = len(result.ids)
durations = [2] * count
durations[0] = durations[-1] = 4                      # BOS/EOS carry the utterance's pauses
frames = sum(durations)
item = {
    "tokens": torch.tensor(result.ids),
    "stress": torch.tensor(result.stress),
    "durations": torch.tensor(durations),
    "speaker": 0,
    "wave": torch.zeros(frames * config.hop, dtype=torch.float32),
    "mel": torch.randn(frames, config.n_mels),
    "prosody": torch.randn(frames, 3),
}
item["pitch_trusted"] = (torch.rand(frames) > 0.25).float()
item["prosody"] = torch.stack((item["pitch_trusted"] * torch.randn(frames) * 0.1,
                               item["pitch_trusted"],
                               torch.full((frames,), config.energy_mean)), dim=-1)
assert int(item["durations"].sum()) == item["mel"].shape[0]
batch = collate([item])
print("batch tokens:", batch["tokens"].shape, "mel:", batch["mel"].shape)
with torch.no_grad():
    loss, terms, _auxiliary = system.objective(batch, 1.0, random_phase=False)
print("objective:", float(loss), "terms:", sorted(terms)[:6])
assert torch.isfinite(loss)
print("PHASE5 MODEL OK  vocab", config.vocab_size, " speakers", config.speakers,
      " params", f"{total:,}", " under 5M:", total < 5_000_000)
