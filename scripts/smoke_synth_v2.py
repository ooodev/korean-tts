"""v2 smoke synthesis: one checkpoint, two serving modes (V2_SPEC.md section 0).

Mode A "Korean speaker": ``ko-loan`` front end, KSS voice — code-mixed Korean with
Korean-style numerals. Mode B "native English": ``own`` front end, F2/M2 voices.
The checkpoint stores the ko-loan front end, so Mode B loads with
``check_frontend=False`` (same 142-token inventory, different conventions).

    TINY_VOCA_FRONTEND=ko-loan DIZIONE_DATA=... python scripts/smoke_synth_v2.py \
        --checkpoint runs/smoke_v2/inference.pt --outdir samples_smoke_v2
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from tiny_voca.infer import load_model, speak
from tiny_voca.text import set_frontend

CASES = [
    # (name, mode, voice, text)
    ("a1_greeting", "A", "KSS", "안녕하세요! 반갑습니다."),
    ("a2_numbers", "A", "KSS", "오늘은 3월 14일이고요, 오후 5시 30분에 만나요."),
    ("a3_codemix", "A", "KSS", "Python을 잘 쓰고 있어요, USB를 연결해서 10km를 뛰었습니다."),
    ("b1_female", "B", "F2", "Good morning, the weather is nice and warm outside."),
    ("b2_male", "B", "M2", "The quick brown fox jumps over the lazy dog."),
    ("b3_money", "B", "F2", "It costs three dollars and fifty cents, can you repeat that?"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("samples_smoke_v2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device, check_frontend=False)
    vocab = model.config.vocab_size
    records = {}
    for name, mode, voice, text in CASES:
        set_frontend("ko-loan" if mode == "A" else "own")
        wave, phones, details = speak(model, text, voice, seed=args.seed)
        if not torch.isfinite(wave).all():
            raise SystemExit(f"{name}: non-finite audio")
        if not 0 < len(wave) or any(t >= vocab for t in phones.ids):
            raise SystemExit(f"{name}: empty audio or out-of-vocab tokens")
        target = args.outdir / f"{name}.wav"
        sf.write(target, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
        records[name] = {
            "mode": mode, "voice": voice, "text": text,
            "tokens": "".join(phones.symbols),
            "seconds": len(wave) / model.config.sample_rate,
            "peak": float(wave.abs().max()),
        }
        print(f"{name:14s} mode {mode} {voice:3s} {records[name]['seconds']:.2f}s "
              f"peak {records[name]['peak']:.4f} -> {target}")
    (args.outdir / "tokens.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE_SYNTH_DONE")


if __name__ == "__main__":
    main()
