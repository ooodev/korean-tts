import argparse
import os
from pathlib import Path
import torch
from .config import CHECKPOINT_FORMAT, Config
from .model import TinyVoca
from .text import FRONTEND_VARIABLE, FRONTENDS, frontend_fingerprint, phonemize, set_frontend


def load_model(path, device="cpu", check_frontend=True):
    """``check_frontend=False`` lets a checkpoint speak tokens of the other
    front end (same inventory, slightly different conventions): for listening
    tests of the new phonemizer before the model is retrained on it."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format: byte-token checkpoints (format 2-3) cannot be used with this code")
    if check_frontend and state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError("checkpoint was trained with a different phoneme inventory/front end")
    model = TinyVoca(Config.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def match_frontend(path):
    """Select the front end the checkpoint was trained with."""
    wanted = torch.load(path, map_location="cpu", weights_only=True)["frontend"]["fingerprint"]
    for name in FRONTENDS:
        set_frontend(name)
        try:
            if frontend_fingerprint() == wanted:
                return name
        except ImportError:
            continue
    set_frontend(FRONTENDS[0])
    return None


def speak(model, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0):
    """Text -> (float waveform tensor, phonemized text). Same front end as training."""
    device = next(model.parameters()).device
    phones = phonemize(text)
    if len(phones) > model.config.max_tokens:
        raise ValueError("text is too long for one utterance; split it into sentences")
    tokens = torch.tensor([phones.ids], device=device)
    stress = torch.tensor([phones.stress], device=device)
    speaker = torch.tensor([model.config.speakers.index(voice)], device=device)
    audio, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)], device=device), speaker, speed, pitch_shift, seed, pitch_range)
    if not torch.isfinite(audio).all():
        raise RuntimeError("model produced non-finite audio")
    return audio[0, :int(lengths[0])].cpu(), phones, details


def main():
    parser = argparse.ArgumentParser(description="Generate speech from a trained tiny-voca inference checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default=None, help="Voice name (default: the checkpoint's first voice)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--pitch-shift", type=float, default=0, help="Added to log F0 (0.058 is about one semitone)")
    parser.add_argument("--pitch-range", type=float, default=1, help="Scale pitch movement around the utterance mean (1 = as predicted)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frontend", choices=FRONTENDS, help="Phonemizer (default: TINY_VOCA_FRONTEND, else the one the checkpoint was trained with)")
    parser.add_argument("--allow-frontend-mismatch", action="store_true",
                        help="Speak although the checkpoint was trained on the other front end's tokens")
    args = parser.parse_args()
    import soundfile as sf
    if args.frontend:
        set_frontend(args.frontend)
    elif FRONTEND_VARIABLE not in os.environ:
        match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, args.device, check_frontend=not args.allow_frontend_mismatch)
    voice = args.voice or model.config.speakers[0]
    if voice not in model.config.speakers:
        raise SystemExit(f"unknown voice {voice!r}; this checkpoint has {list(model.config.speakers)}")
    wave, phones, _ = speak(model, args.text, voice, args.speed, args.pitch_shift, pitch_range=args.pitch_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Floating WAV preserves output for inspection; no silent clipping or normalization.
    sf.write(args.output, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
    print(" ".join(phones.symbols))
    print(f"Saved {len(wave)/model.config.sample_rate:.2f}s; peak={float(wave.abs().max()):.4f}")


if __name__ == "__main__":
    main()
