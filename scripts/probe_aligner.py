"""Probe the Korean CTC recogniser that Phase 5 will use as the aligner.

Prints the label inventory, the frame grid, and how well a greedy decode of a
real KSS clip covers the tokens the Korean front end produced.
"""

import os
import sys
import unicodedata

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")
os.environ.setdefault("DIZIONE_DATA", r"C:\ai-workspace\korean-tts\weights\phonemizer")

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

MODEL = "kresnik/wav2vec2-large-xlsr-korean"
HOP, RATE = 256, 24000

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(device).eval()
labels = processor.tokenizer.get_vocab()
print("vocab size:", len(labels), "blank id:", processor.tokenizer.pad_token_id)
print("first labels:", list(labels)[:12])

dataset = load_dataset("Bingsu/KSS_Dataset", split="train").cast_column("audio", Audio(decode=False))
print("rows:", len(dataset))

from tiny_voca import text

for index in (0, 7, 123):
    row = dataset[index]
    audio = row["audio"]
    print("-" * 70)
    print("audio keys:", {k: type(v).__name__ for k, v in audio.items()})
    if audio.get("bytes") is not None:
        import io
        wave, rate = sf.read(io.BytesIO(audio["bytes"]))
    else:
        wave, rate = sf.read(audio["path"])
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if rate != 16000:
        from scipy.signal import resample_poly
        wave = resample_poly(wave, 160, rate // 100).astype(np.float32)
        rate = 16000
    print("-" * 70)
    print("row", index, "sr", rate, "samples", len(wave), "dur", round(len(wave) / rate, 3))
    print("text:", row["expanded_script"])
    phonemized = text.phonemize(row["expanded_script"])
    print("tokens:", "".join(phonemized.symbols).replace("<sp>", "|"))
    print("n tokens:", len(phonemized.symbols), "units:", len(phonemized.units))
    inputs = processor(wave.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    ids = logits.argmax(-1)[0]
    decoded = processor.batch_decode(ids.unsqueeze(0))[0]
    frames = ids.shape[0]
    print("ctc frames:", frames, "ms/frame:", round(1000 * 16000 / frames / (len(wave) / 16000), 3))
    print("decoded:", decoded)
    spoken = [s for s in phonemized.symbols if unicodedata.category(s).startswith("L") or len(s) == 1]
    print("syllables in text:", sum(1 for s in phonemized.symbols if 0xAC00 <= ord(s) <= 0xD7A3))
