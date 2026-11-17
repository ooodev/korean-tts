"""Final v2 listening set: one checkpoint, both serving modes (V2_SPEC section 0).

Mode A "Korean speaker": ``ko-loan`` front end, KSS voice — Korean with Korean-style
numerals and code-mixed English words. Mode B "native English": ``own`` front end,
F2/M2 voices — English with English numerals. The checkpoint stores the ko-loan
front end, so the model is loaded with ``check_frontend=False`` and the front end
is switched per case (``own``'s 76 tokens are the prefix of the unified inventory).

    TINY_VOCA_FRONTEND=ko-loan DIZIONE_DATA=... python scripts/listening_v2.py \
        --checkpoint runs/v2/inference.pt --outdir samples_v2 --device cuda

Writes one wav per case plus tokens.json. Intended for the human listening test
that every gate number supports; run it on the final checkpoint and on the
gate checkpoints if a listening pass is wanted there.
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
    # --- Mode A: Korean, KSS voice ---
    ("a1_greeting", "A", "KSS", "안녕하세요! 반갑습니다."),
    ("a2_numbers", "A", "KSS", "오늘은 3월 14일이고요, 오후 5시 30분에 만나요."),
    ("a3_codemix", "A", "KSS", "Python을 잘 쓰고 있어요. USB를 연결해서 10km를 뛰었습니다."),
    ("a4_flow", "A", "KSS", "저녁에는 가족들과 함께 저녁 식사를 하고, 그 다음에 산책을 하려고 합니다."),
    ("a5_question", "A", "KSS", "지금 몇 시인지 혹시 알려주실 수 있나요?"),
    ("a6_exclaim", "A", "KSS", "와, 정말 대단하네요! 드디어 한국어 합성에 성공했습니다!"),
    ("a7_hardconsonants", "A", "KSS", "큰집 건너편에서 깍두기를 깎아 먹었습니다."),
    ("a8_numbers_units", "A", "KSS", "물 2리터와 빵 3개를 사서 집에 돌아왔습니다."),
    # --- Mode B: English, F2/M2 voices ---
    ("b1_female_greeting", "B", "F2", "Good morning, the weather is nice and warm outside."),
    ("b2_male_dates", "B", "M2", "It's March fourteenth, and the meeting starts at five thirty."),
    ("b3_female_money", "B", "F2", "The total comes to thirty-seven point five dollars."),
    ("b4_male_flow", "B", "M2", "I walked to the store, bought some bread and milk, and came back home before the rain started."),
    ("b5_female_question", "B", "F2", "Could you tell me where the nearest subway station is?"),
    ("b6_male_codemix", "B", "M2", "I really enjoy working with the Python framework and the new USB device."),
    ("b7_female_stress", "B", "F2", "The weather was wonderful, so we went out and stayed until the very end."),
    ("b8_male_numbers", "B", "M2", "Please dial one twenty-three, then press four twice."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("samples_v2"))
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
        if not len(wave) or any(t >= vocab for t in phones.ids):
            raise SystemExit(f"{name}: empty audio or out-of-vocab tokens")
        target = args.outdir / f"{name}.wav"
        sf.write(target, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
        records[name] = {
            "mode": mode, "voice": voice, "text": text,
            "tokens": "".join(phones.symbols),
            "seconds": len(wave) / model.config.sample_rate,
            "peak": float(wave.abs().max()),
        }
        print(f"{name:18s} mode {mode} {voice:3s} {records[name]['seconds']:.2f}s "
              f"peak {records[name]['peak']:.4f} -> {target}")
    (args.outdir / "tokens.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LISTENING_V2_DONE")


if __name__ == "__main__":
    main()
