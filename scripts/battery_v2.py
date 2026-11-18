"""V2_SPEC section 5 listening batteries: loanwords and numbers, Mode A only.

Imports the P1 golden tables (``tiny-voca/tests/test_frontend_gate.py``) so the
expected readings are the locked contract, re-checks them at the text level
(P1 gate discipline), then synthesises each case in Mode A (KSS, ``ko-loan``)
with optional Whisper-KO ASR as an automatic proxy for the human listening
test. The spec gates: loanword battery 20 words / 100 % correct syllables
(listening); number battery 20 cases / 100 % correct reading (listening + ASR).

    TINY_VOCA_FRONTEND=ko-loan DIZIONE_DATA=... python scripts/battery_v2.py \
        --checkpoint runs/v2/inference.pt --outdir samples_v2_battery --device cuda

Writes one wav per case, plus battery_summary.json with text-level results
and ASR transcripts/CERs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tiny-voca" / "tests"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from test_frontend_gate import LOANWORD_CASES, TN_CASES  # noqa: E402
from tiny_voca.infer import load_model, speak  # noqa: E402
from tiny_voca.korean import korean_normalize, phonemize_mixed  # noqa: E402
from tiny_voca.text import set_frontend  # noqa: E402

# Spec: number battery = counters / phones / dates / percent (20 cases).
NUMBER_INPUTS = [
    "3개", "3명", "3살", "3번째", "20살", "3세", "123번", "402호",      # counters/codes
    "010-1234-5678",                                                     # phone
    "3시 30분", "14:30", "2026년 3월 14일", "6월", "3/14", "100~200명",  # dates/time/range
    "5%", "$12.99", "3.5", "10km",                                       # percent/currency/decimal/unit
]

# Spec: loanword battery = 20 words (everyday, tech, long, acronyms).
LOANWORD_INPUTS = [
    "hello", "internet", "computer", "office", "dollar", "python",
    "coffee", "algorithm", "chocolate", "digital", "bottle", "clock",
    "gymnasium", "nameplate", "xylophone", "smart", "bus", "bag",
    "USB", "AI",
]

TN_LOOKUP = dict(TN_CASES)
LOAN_LOOKUP = dict(LOANWORD_CASES)


def units_of(text: str):
    """Mode A phonemisation -> joined Hangul syllables (the listenable units).

    ``Phonemized.units`` is a tuple of ``(label, start, end)`` triples where the
    label is the alignable syllable/token (the ids indices are irrelevant here).
    """
    out = phonemize_mixed(text, "ko-loan")
    return "".join(label for label, _s, _e in out.units if label) if out.units else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("samples_v2_battery"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-asr", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    number_cases = [(i, TN_LOOKUP[i]) for i in NUMBER_INPUTS]
    loanword_cases = [(w, LOAN_LOOKUP[w]) for w in LOANWORD_INPUTS]

    # --- text-level pre-check: the locked readings must hold (P1 discipline) ---
    text_results = []
    for battery, cases in (("number", number_cases), ("loanword", loanword_cases)):
        for src, expected in cases:
            if battery == "number":
                actual = korean_normalize(src)
            else:
                actual = units_of(src)
            text_results.append({"battery": battery, "input": src, "expected": expected,
                                 "actual": actual, "ok": actual == expected})
    bad = [r for r in text_results if not r["ok"]]
    if bad:
        for r in bad:
            print(f"TEXT-LEVEL MISMATCH {r['battery']}: {r['input']} expected {r['expected']} got {r['actual']}")
        raise SystemExit("front end no longer matches the locked readings; do not run the audio battery")

    # --- synthesis + optional ASR ---
    model = load_model(args.checkpoint, args.device, check_frontend=False)
    asr, transcribe = None, None
    if not args.no_asr:
        import jiwer
        from faster_whisper import WhisperModel
        device = "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        asr = WhisperModel("small", device=device,
                           compute_type="float16" if device == "cuda" else "int8", cpu_threads=8)

        def transcribe(path):
            from scipy.signal import resample_poly
            import numpy as np
            audio, rate = sf.read(path, dtype="float32")
            g = int(np.gcd(rate, 16000))
            audio = resample_poly(audio, 16000 // g, rate // g).astype("float32")
            segments, _ = asr.transcribe(audio, language="ko", beam_size=5,
                                         condition_on_previous_text=False, temperature=0.0)
            return " ".join(s.text.strip() for s in segments).strip()

    set_frontend("ko-loan")
    records = []
    for r in text_results:
        # Deterministic, filesystem-safe case name: a blake2b digest plus the
        # input with any path-hostile characters replaced (e.g. "3/14" -> "3_14").
        import hashlib
        digest = hashlib.blake2b(f"{r['battery']}|{r['input']}".encode("utf-8"),
                                 digest_size=2).hexdigest()
        safe_input = "".join(ch if (ch.isalnum() or ch in "-_.") else "_"
                             for ch in r["input"][:12])
        name = f"{r['battery']}_{digest}_{safe_input}"
        wave, phones, _ = speak(model, r["input"], "KSS", seed=args.seed)
        target = args.outdir / f"{name}.wav"
        sf.write(target, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
        rec = {**r, "file": target.name, "seconds": round(len(wave) / model.config.sample_rate, 3),
               "peak": round(float(wave.abs().max()), 4)}
        if asr is not None:
            hyp = transcribe(target)
            rec["asr_transcript"] = hyp
            rec["asr_cer"] = round(float(jiwer.cer(r["expected"].replace(" ", ""),
                                                   hyp.replace(" ", "") or "<empty>")), 4)
        records.append(rec)
        print(f"{name} {r['input']!r} -> {r['expected']!r} {rec['seconds']:.2f}s"
              + (f" asr={rec['asr_transcript']!r} cer={rec['asr_cer']}" if asr is not None else ""))

    n = len(records)
    asr_cers = [r["asr_cer"] for r in records if "asr_cer" in r]
    summary = {"checkpoint": str(args.checkpoint), "cases": n,
               "text_level_ok": sum(r["ok"] for r in records),
               "asr_mean_cer": round(sum(asr_cers) / len(asr_cers), 4) if asr_cers else None,
               "records": records}
    (args.outdir / "battery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("BATTERY_V2_DONE")


if __name__ == "__main__":
    main()
