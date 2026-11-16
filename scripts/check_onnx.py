"""Gate: the exported ONNX graph must speak like the torch model.

    python scripts/check_onnx.py --checkpoint runs/full/inference.pt \
        --onnx tiny-voca/web/public/assets/model.onnx \
        --strict runs/full/model-noise.onnx

Two graphs, two checks. The graph the browser loads generates its own noise for
the source-filter head, so its waveform is a different realisation of the same
voice: durations must be identical and the correlation high. The strict graph is
the same network exported with ``--external-noise``; fed the torch noise it must
reproduce the waveform sample for sample, which is what proves the export is
faithful rather than merely similar.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")

import numpy as np
import onnxruntime as ort
import torch

from tiny_voca.infer import load_model
from tiny_voca.text import phonemize

SENTENCES = [
    "안녕하세요. 오늘 날씨가 정말 맑고 좋네요.",
    "지금 몇 시인지 혹시 알려주실 수 있나요?",
    "이천이십육 년 십일 월 십오 일 오후 세 시 삼십 분입니다.",
    "오늘 Python으로 테스트를 진행했습니다.",
    "와, 정말 대단하네요! 드디어 한국어 합성에 성공했습니다!",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/full/inference.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("tiny-voca/web/public/assets/model.onnx"))
    parser.add_argument("--strict", type=Path, default=Path("runs/full/model-noise.onnx"),
                        help="graph exported with --external-noise, for the exact check")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.checkpoint, "cpu")
    voice = args.voice or model.config.speakers[0]
    graph = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    strict = ort.InferenceSession(str(args.strict), providers=["CPUExecutionProvider"])
    speaker = model.config.speakers.index(voice)

    print(f"model {args.checkpoint} voice {voice} | graph {args.onnx.name} "
          f"({args.onnx.stat().st_size / 1e6:.1f} MB) | strict {args.strict.name}")
    print(f"{'sentence':<34} {'tokens':>6} {'seconds':>7} {'dur':>4} {'corr':>7} {'peak':>8}")

    worst = {"duration": 0, "corr": 1.0, "peak": 0.0, "strict": 0.0}
    for text in SENTENCES:
        phones = phonemize(text)
        tokens = torch.tensor([phones.ids], dtype=torch.long)
        stress = torch.tensor([phones.stress], dtype=torch.long)
        lengths = torch.tensor([len(phones)], dtype=torch.long)
        audio, frame_lengths, details = model.synthesize(
            tokens, stress, lengths, torch.tensor([speaker], dtype=torch.long),
            args.speed, seed=args.seed)
        torch_durations = details["durations"][0, :len(phones)].numpy()

        feed = {
            "tokens": tokens.numpy().astype("int64"),
            "stress": stress.numpy().astype("int64"),
            "voice": np.array([speaker], dtype="int64"),
            "speed": np.array([args.speed], dtype="float32"),
            "pitch_shift": np.array([0.0], dtype="float32"),
            "pitch_range": np.array([1.0], dtype="float32"),
        }
        out = graph.run(None, feed)
        duration_error = int(np.abs(torch_durations - out[1][0][:len(phones)]).max())
        samples = min(audio.shape[-1], out[0].shape[-1])
        a = audio[0, :samples].numpy()
        b = out[0][0, :samples]
        corr = float(np.corrcoef(a, b)[0, 1])
        peak = float(np.abs(a - b).max())

        # the same graph with the torch noise stream: must be bit-close
        noise = torch.randn(int(frame_lengths[0]) * model.config.hop,
                            generator=torch.Generator().manual_seed(args.seed))
        exact = strict.run(None, {**feed, "noise": noise[None].numpy()})
        n = min(audio.shape[-1], exact[0].shape[-1])
        strict_peak = float(np.abs(audio[0, :n].numpy() - exact[0][0, :n]).max())

        worst["duration"] = max(worst["duration"], duration_error)
        worst["corr"] = min(worst["corr"], corr)
        worst["peak"] = max(worst["peak"], peak)
        worst["strict"] = max(worst["strict"], strict_peak)
        label = text if len(text) <= 32 else text[:31] + "…"
        print(f"{label:<34} {len(phones):>6} {audio.shape[-1] / model.config.sample_rate:>7.2f} "
              f"{duration_error:>4} {corr:>7.4f} {peak:>8.4f}  strict peak {strict_peak:.5f}")

    ok = worst["duration"] == 0 and worst["corr"] > 0.98 and worst["strict"] < 2e-3
    print(f"\ndurations exact: {worst['duration'] == 0} | min correlation {worst['corr']:.4f} "
          f"(noise realisation differs, peak {worst['peak']:.3f}) | "
          f"strict graph peak {worst['strict']:.6f} (~{20 * np.log10(max(worst['strict'], 1e-9)):.0f} dB, "
          f"float32 conv accumulation) -> {'OK' if ok else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
