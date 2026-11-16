"""Build the pronunciation dictionary from CMUdict. Cloud only.

    python build_lexicon.py --cmudict data/cmudict/cmudict.dict --output built

Writes
    <output>/align.json     which letters spell which phoneme (EM over CMUdict)
    <output>/lexicon.tsv    word <TAB> phones of the TTS inventory, stress marked
    <output>/lexicon.json   counts and source hash

Only CMUdict goes in. The conversion to the TTS phone inventory is the rule
set in ``dizione.convert``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dizione import align, convert      # noqa: E402

_WORD = re.compile(r"^[a-z]+(?:'[a-z]+)*'?$|^'[a-z]+$")
_CLOTH = ("F", "S", "TH", "NG")


def read_cmudict(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        word = re.sub(r"\(\d+\)$", "", word)
        variants = entries.setdefault(word, [])
        if phones and phones not in variants:
            variants.append(phones)
    return entries


def choose(word, variants):
    """The pronunciation that goes into the lexicon: CMUdict's first entry,
    unless that entry merges two vowels the TTS inventory keeps apart and
    another entry does not. Where CMUdict lists both AA and AO for one vowel,
    the AO entry is taken for the THOUGHT set (caught, bought, law: written
    au, aw, augh, ough) and for the CLOTH set (soft, cost, coffee: "o" before
    f, s, th, ng), as the unambiguous words of those sets have it."""
    first = variants[0]
    for other in variants[1:]:
        if len(other) != len(first):
            continue
        differences = [i for i, (a, b) in enumerate(zip(first, other)) if a != b]
        if len(differences) != 1:
            continue
        i = differences[0]
        if first[i][:2] != "AA" or other[i][:2] != "AO" or first[i][2:] != other[i][2:]:
            continue
        thought = any(spelling in word for spelling in ("au", "aw", "ough"))
        cloth = not thought and "o" in word and i + 1 < len(first) and first[i + 1] in _CLOTH
        if thought or cloth:
            return other
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-alignment", action="store_true", help="Keep an existing align.json (rule changes only)")
    args = parser.parse_args()
    started = time.time()
    log = lambda message: print(json.dumps({"elapsed": round(time.time() - started, 1), "message": message}), flush=True)

    entries = {word: variants for word, variants in read_cmudict(args.cmudict).items() if _WORD.match(word)}
    args.output.mkdir(parents=True, exist_ok=True)
    table_path = args.output / "align.json"
    if args.reuse_alignment and table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
    else:
        pairs = [(word, variant) for word, variants in entries.items() for variant in variants]
        table = align.train(pairs, args.iterations, args.workers, log)
        table_path.write_text(json.dumps(table, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    aligner = align.Aligner(table)

    lines, unaligned = [], 0
    for word in sorted(entries):
        chosen = choose(word, entries[word])
        runs = aligner.align(word, chosen)
        unaligned += runs is None
        lines.append(word + "\t" + convert.to_text(convert.convert(word, chosen, runs)) + "\n")
    (args.output / "lexicon.tsv").write_text("".join(lines), encoding="utf-8")
    summary = {"words": len(lines), "without_alignment": unaligned, "alignment_pairs": sum(len(v) for v in table.values()),
               "cmudict_sha256": hashlib.sha256(args.cmudict.read_bytes()).hexdigest(),
               "lexicon_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}
    (args.output / "lexicon.json").write_text(json.dumps(summary, indent=1))
    log(summary)


if __name__ == "__main__":
    main()
