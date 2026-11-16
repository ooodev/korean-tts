"""Programmatic audio checks for generated .wav files (Phase 6 / Phase 8 gate).

    python scripts/verify_wavs.py eval/smoke/smoke_01.wav eval/smoke/smoke_02.wav

Checks: exists, sample rate, mono, duration window, no NaN/Inf, RMS in a healthy
range, peak not clipping. Exits non-zero when any file fails.
"""
import glob
import sys

import numpy as np
import soundfile as sf

MIN_DURATION = 1.0
MAX_DURATION = 15.0
MIN_RMS = 0.02


def check(path, min_duration=MIN_DURATION, max_duration=MAX_DURATION):
    samples, rate = sf.read(path, always_2d=True)
    channels = samples.shape[1]
    mono = samples.mean(axis=1)
    duration = len(mono) / rate
    finite = bool(np.isfinite(mono).all())
    rms = float(np.sqrt(np.mean(mono ** 2))) if finite else 0.0
    peak = float(np.max(np.abs(mono))) if finite else 0.0
    problems = []
    if rate != 24000:
        problems.append(f"sample rate {rate}")
    if channels != 1:
        problems.append(f"{channels} channels")
    if not min_duration < duration < max_duration:
        problems.append(f"duration {duration:.2f}s outside ({min_duration}, {max_duration})")
    if not finite:
        problems.append("non-finite samples")
    if rms < MIN_RMS:
        problems.append(f"rms {rms:.4f} too quiet")
    if peak > 1.0:
        problems.append(f"peak {peak:.3f} clips")
    print(f"{path:44s} {rate} Hz  {channels}ch  {duration:6.2f}s  rms {rms:.4f}  peak {peak:.4f}  "
          f"{'OK' if not problems else 'FAIL ' + '; '.join(problems)}")
    return not problems


if __name__ == "__main__":
    arguments = sys.argv[1:]
    limits = {}
    for flag, key in (("--min-duration", "min_duration"), ("--max-duration", "max_duration")):
        if flag in arguments:
            index = arguments.index(flag)
            limits[key] = float(arguments[index + 1])
            del arguments[index:index + 2]
    paths = [match for argument in arguments for match in sorted(glob.glob(argument)) or [argument]]
    raise SystemExit(0 if all(check(path, **limits) for path in paths) else 1)
