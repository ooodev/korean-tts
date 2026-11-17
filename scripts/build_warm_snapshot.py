"""Build the D1 warm-start snapshot: v1 generator weights on the v2 recipe.

V2_SPEC D1: if Korean CER > 12.5 % at the 120k checkpoint of the from-scratch
run, abandon the v2 weights and continue from v1's converged generator
(``runs/full/inference.pt``) on the v2 bilingual data. ``train.py --init-from``
needs a FULL training-format checkpoint (``system`` dict, critics, optimizers,
recipe), and its strict config check requires the embedded config to equal the
v2 training config — so this rebuilds that snapshot:

* ``system``: v1 generator weights under the ``model.`` prefix, stored as-is;
  ``train.py``'s ``load_compatible`` then places them into the v2 model (the
  speaker embedding grows 1->3 rows into its leading block; mel_head is fresh),
* ``critics`` / ``critic_optimizer``: freshly initialised for the v2 config —
  train.py's branch logic takes trained critics over when step > gan_start, so
  these are what the warm run inherits; GAN scaling tolerates them,
* ``config``: config-v2, ``step``: 130000 (v1's final step), v1's
  frontend/annotation fingerprints recorded (the run is started with
  ``--new-data``).

    python scripts/build_warm_snapshot.py \
        --source runs/full/inference.pt --config config-v2.json \
        --output eval/v2/warm-snapshot.pt

The warm run then starts with:
    train.py --init-from eval/v2/warm-snapshot.pt --new-data --output runs/v2-warm ...
"""
import argparse
import json
from pathlib import Path

import torch

from tiny_voca.config import CHECKPOINT_FORMAT, Config
from tiny_voca.train import Discriminators, TrainingSystem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="v1 slim inference checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="v2 training config (config-v2.json)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=130000, help="v1's final step (the warm run continues from here)")
    # Must mirror tiny_voca.train's optimizer construction (train.py:356-357).
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    v1 = torch.load(args.source, map_location="cpu", weights_only=True)
    if v1.get("format_version") != CHECKPOINT_FORMAT:
        raise SystemExit("source is not a slim inference checkpoint")
    config = Config.from_dict(json.loads(args.config.read_text(encoding="utf-8")))

    # Dry-run the expected load_compatible report (without loading): which v1
    # tensors grow, which v2 tensors are fresh, and that nothing shrank.
    system = TrainingSystem(config)
    expected = system.state_dict()
    system_state = {f"model.{key}": value for key, value in v1["model"].items()}
    report = {"exact": 0, "grown": [], "new": [], "dropped": []}
    for name, tensor in expected.items():
        if name not in system_state:
            report["new"].append(name)
        elif system_state[name].shape == tensor.shape:
            report["exact"] += 1
        elif system_state[name].ndim == tensor.ndim and all(a <= b for a, b in zip(system_state[name].shape, tensor.shape)):
            report["grown"].append(f"{name} {tuple(system_state[name].shape)} -> {tuple(tensor.shape)}")
        else:
            raise SystemExit(f"source tensor cannot branch: {name} "
                             f"{tuple(system_state[name].shape)} !> {tuple(tensor.shape)}")
    report["dropped"] = sorted(set(system_state) - set(expected))
    if report["dropped"]:
        raise SystemExit(f"source tensors unknown to the v2 model: {report['dropped']}")

    # Fresh critics + optimizer moments for the exact objects train.py creates.
    critics = Discriminators()
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=0.0)

    # v2 training recipe, mirroring scripts/train_full.ps1 defaults.
    recipe = {"batch_frames": 12000, "max_batch": 48, "learning_rate": args.lr, "warmup": 1000,
              "lr_half_life": 100000, "grad_clip": None, "gan_start": 8000, "gan_ratio": 0.7,
              "gan_ramp": 3000, "feature_matching": True, "gan_scale_max": None, "crop_samples": None,
              "crops": None, "teacher_hold": 3000, "teacher_ramp": 7000, "teacher_final": 0.0,
              "subset": None, "validation_size": 32, "seed": None}

    snapshot = {
        "format_version": CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "frontend": v1["frontend"],
        "annotation": v1.get("annotation"),
        "step": args.step,
        "datasets": [],
        "recipe": recipe,
        "system": system_state,
        "critics": critics.state_dict(),
        "optimizer": optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "sampler": {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, args.output)

    print(json.dumps({"snapshot": str(args.output), "step": args.step,
                      "source_model_params": sum(v.numel() for v in v1["model"].values()),
                      "expected_load": report,
                      "config_speakers": list(config.speakers),
                      "note": "load with --init-from --new-data; grown tensors get leading-block placement by load_compatible"},
                     indent=1))


if __name__ == "__main__":
    main()
