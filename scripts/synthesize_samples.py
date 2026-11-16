"""Synthesize every utterance of a manifest with a trained checkpoint (Phase 8).

    python scripts/synthesize_samples.py --checkpoint runs/full/inference.pt \
           --manifest samples/manifest.json --outdir samples --device cuda

Writes ``samples/<name>.wav`` plus ``samples/tokens.json`` (the exact token
sequence the model saw, which is what the code-mixed sample is judged on).
"""
import argparse
import json
from pathlib import Path

import soundfile as sf

from tiny_voca.infer import load_model, speak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch-range", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    model = load_model(arguments.checkpoint, arguments.device)
    voice = arguments.voice or model.config.speakers[0]
    cases = json.load(open(arguments.manifest, encoding="utf-8"))
    arguments.outdir.mkdir(parents=True, exist_ok=True)

    tokens = {}
    for case in cases:
        wave, phones, _ = speak(model, case["text"], voice, arguments.speed,
                                pitch_range=arguments.pitch_range, seed=arguments.seed)
        target = arguments.outdir / f"{case['name']}.wav"
        sf.write(target, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
        tokens[case["name"]] = {
            "text": case["text"],
            "tokens": "".join(phones.symbols),
            "units": [[label, start, stop] for label, start, stop in phones.units],
            "seconds": len(wave) / model.config.sample_rate,
            "peak": float(wave.abs().max()),
        }
        print(f"{case['name']:28s} {tokens[case['name']]['seconds']:.2f}s "
              f"peak {tokens[case['name']]['peak']:.4f} -> {target}")

    with open(arguments.outdir / "tokens.json", "w", encoding="utf-8") as handle:
        json.dump(tokens, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
