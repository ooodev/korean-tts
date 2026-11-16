"""Korean ASR back-transcription check for synthesized samples (Phase 8 gate).

Transcribes each .wav with faster-whisper and compares the hypothesis with the
reference text. Korean is measured at character level (CER, spaces ignored) and
at word level (WER on whitespace-separated eojeols).

    python scripts/asr_check.py --model small --manifest samples/manifest.json
    python scripts/asr_check.py --model small samples/a.wav "안녕하세요 ..." samples/b.wav "..."
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import torch  # noqa: F401  its lib directory carries the cuBLAS DLLs ctranslate2 loads

_TORCH_LIB = str(Path(torch.__file__).resolve().parent / "lib")
if hasattr(os, "add_dll_directory"):          # Windows: ctranslate2 finds cublas64_12.dll here
    os.add_dll_directory(_TORCH_LIB)
os.environ["PATH"] = os.pathsep.join([_TORCH_LIB, os.environ.get("PATH", "")])

from faster_whisper import WhisperModel  # noqa: E402

PUNCTUATION = ".,!?;:\"'()[]…·"
# Whisper writes Korean numbers as Arabic digits ("이천이십육 년" -> "2026 년"), which a
# character-level score counts as several substitutions. Folding a digit run and a run of
# Sino-Korean numeral morphemes to the same placeholder measures the reading, not the
# orthography; it is reported next to the raw CER, never instead of it.
NUMERAL_MORPHEMES = "영공일이사오육칠팔구십백천만억점시분초년월일"


def fold_numerals(text):
    return re.sub("[" + NUMERAL_MORPHEMES + "]+", "#", re.sub(r"[0-9]+", "#", text))


def normalize(text):
    """NFC, drop punctuation, collapse whitespace: the ASR comparison alphabet."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if char not in PUNCTUATION)
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    """Levenshtein distance between two token lists, two rolling rows."""
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        rows = [i]
        for j, spoken in enumerate(hypothesis, 1):
            cost = 0 if spoken == expected else 1
            rows.append(min(rows[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = rows
    return previous[-1]


def scores(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    cer = _cer(reference, hypothesis)
    words, heard = reference.split(), hypothesis.split()
    wer = edit_distance(words, heard) / max(1, len(words))
    return cer, wer, _cer(fold_numerals(reference), fold_numerals(hypothesis))


def _cer(reference, hypothesis):
    spoken = [char for char in hypothesis if char != " "]
    characters = [char for char in reference if char != " "]
    return edit_distance(characters, spoken) / max(1, len(characters))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="*", help="wav path followed by its reference text")
    parser.add_argument("--manifest", help="JSON list of {name, text[, wav]} objects")
    parser.add_argument("--outdir", type=Path, help="Where <name>.wav lives for manifest rows")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", dest="as_json", help="Write the table to this JSON file")
    arguments = parser.parse_args()

    cases = []
    if arguments.manifest:
        for row in json.load(open(arguments.manifest, encoding="utf-8")):
            path = row.get("wav") or (arguments.outdir / f"{row['name']}.wav" if arguments.outdir
                                      else Path(f"{row['name']}.wav"))
            cases.append((str(path), row["text"], row.get("name", str(path))))
    for path, text in zip(arguments.pairs[0::2], arguments.pairs[1::2]):
        cases.append((path, text, path))

    model = WhisperModel(arguments.model, device=arguments.device, compute_type="float16")
    rows = []
    for path, text, name in cases:
        segments, _ = model.transcribe(path, language="ko", beam_size=5, vad_filter=False)
        hypothesis = "".join(segment.text for segment in segments)
        cer, wer, cer_numbers = scores(text, hypothesis)
        rows.append({"name": name, "wav": path, "reference": normalize(text),
                     "hypothesis": normalize(hypothesis), "cer": cer, "wer": wer,
                     "cer_numbers_folded": cer_numbers})
        print(f"--- {name}\n  ref : {normalize(text)}\n  asr : {normalize(hypothesis)}\n"
              f"  CER {cer * 100:.1f} %   WER {wer * 100:.1f} %   CER (numbers folded) {cer_numbers * 100:.1f} %")

    if rows:
        print("\nmean CER %.1f %%  mean WER %.1f %%  mean CER (numbers folded) %.1f %%"
              % (100 * sum(r["cer"] for r in rows) / len(rows), 100 * sum(r["wer"] for r in rows) / len(rows),
                 100 * sum(r["cer_numbers_folded"] for r in rows) / len(rows)))
    if arguments.as_json:
        with open(arguments.as_json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
