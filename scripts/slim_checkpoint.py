"""Slim a training checkpoint (``step-*.pt`` / ``last.pt``) to inference format.

``tiny_voca.train`` saves full training checkpoints (``system`` dict with critics,
optimizers and RNG state) as step-*.pt, and a slim copy (``model`` dict) as
inference.pt only at chunk boundaries. To gate a step-N checkpoint without
waiting for the next chunk boundary, rebuild the slim format:

    python scripts/slim_checkpoint.py --input runs/v2/step-00080000.pt \
        --output eval/v2/step-80000/inference.pt
"""
import argparse
import os
import tempfile
import torch

from tiny_voca.config import CHECKPOINT_FORMAT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise SystemExit("unsupported checkpoint format")
    prefix = "model."
    model_state = {k[len(prefix):]: v for k, v in state["system"].items() if k.startswith(prefix)}
    if not model_state:
        raise SystemExit("no generator weights found (expected a 'system' dict with model.* keys)")
    slim = {
        "format_version": CHECKPOINT_FORMAT,
        "config": state["config"],
        "step": state["step"],
        "frontend": state["frontend"],
        "annotation": state.get("annotation"),
        "model": model_state,
    }
    out = os.fspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(out) or ".", delete=False, suffix=".tmp") as tmp:
        tmp_path = tmp.name
    torch.save(slim, tmp_path)
    os.replace(tmp_path, out)
    total = sum(v.numel() for v in model_state.values())
    finite = all(torch.isfinite(v).all() for v in model_state.values())
    print(f"slimmed {args.input} -> {out} | step {state['step']} | "
          f"{len(model_state)} tensors, {total:,} params | finite={finite}")


if __name__ == "__main__":
    main()
