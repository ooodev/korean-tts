"""Synthesize one probe sentence from every checkpoint of a run (training audit).

``tiny_voca.train`` writes metrics and checkpoints but no audio, so this walks
``runs/<name>/step-*.pt`` (and ``last.pt``) and speaks the same sentence from
each, which makes the progress audible: ``eval/probe/<step>.wav``.

    python scripts/checkpoint_probe.py --run runs/full --device cuda
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from tiny_voca.config import Config
from tiny_voca.model import TinyVoca
from tiny_voca.infer import speak

PROBE = "오늘 날씨가 정말 맑고 좋네요. 저녁에 같이 산책할까요?"


def load_step(path, device):
    """A training checkpoint holds the whole system; keep only the generator."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyVoca(Config.from_dict(state["config"]))
    prefix = "model."
    model.load_state_dict({key[len(prefix):]: value for key, value in state["system"].items()
                           if key.startswith(prefix)})
    return model.to(device).eval(), state["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text", default=PROBE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--every", type=int, default=1, help="Speak one of every N checkpoints")
    arguments = parser.parse_args()

    output = arguments.output or arguments.run.parent / f"probe-{arguments.run.name}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(arguments.run.glob("step-*.pt")) + [arguments.run / "last.pt"]
    seen_steps = set()
    for position, path in enumerate(checkpoints):
        if not path.is_file() or position % arguments.every:
            continue
        model, step = load_step(path, arguments.device)
        if step in seen_steps:          # last.pt is usually the newest step-*.pt
            continue
        seen_steps.add(step)
        wave, phones, _ = speak(model, arguments.text, model.config.speakers[0])
        target = output / f"step-{step:08d}.wav"
        sf.write(target, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
        print(json.dumps({"step": step, "wav": str(target),
                          "seconds": round(len(wave) / model.config.sample_rate, 3),
                          "peak": round(float(wave.abs().max()), 4),
                          "tokens": "".join(phones.symbols)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
