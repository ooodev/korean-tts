"""Train the homograph classifiers. Cloud only.

    python train_homographs.py --data data/homograph/data --cmudict data/cmudict/cmudict.dict \
        --built built

Source: Wikipedia Homograph Data (Google, Apache-2.0): 162 homographs, about
100 labelled sentences each, with a suggested 90/10 split. Training uses the
train files only; the eval files give the reported accuracy. Only the weights
are written to ``<built>/homographs.json``, never the sentences.

Pronunciation of each sense: the CMUdict entry closest to the pronunciation in
``wordids.tsv``, corrected by ``data/homograph_fixes.txt``, then converted by
the same rules as the dictionary.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dizione import align, convert, tables            # noqa: E402
from dizione.homographs import TOKEN, features, sentence_of, shared      # noqa: E402

_MULTI = ("ɑː", "iː", "uː", "ɔː", "oʊ", "aɪ", "aʊ", "eɪ", "ɔɪ")
_VOWELS = {"æ": "AE", "ɑː": "AA", "ə": "AH", "ɛ": "EH", "ɪ": "IH", "iː": "IY", "oʊ": "OW", "uː": "UW", "ʊ": "UH", "ʌ": "AH",
           "ɔː": "AO", "aɪ": "AY", "aʊ": "AW", "eɪ": "EY", "ɔɪ": "OY", "ɚ": "ER", "ɔ": "AO", "ɑ": "AA", "i": "IY", "u": "UW"}
_CONSONANTS = {"b": "B", "d": "D", "f": "F", "ɡ": "G", "g": "G", "h": "HH", "k": "K", "l": "L", "m": "M", "n": "N", "ŋ": "NG",
               "p": "P", "ɹ": "R", "s": "S", "t": "T", "v": "V", "w": "W", "z": "Z", "ʃ": "SH", "ʒ": "ZH", "θ": "TH", "ð": "DH",
               "j": "Y", "ʧ": "CH", "ʤ": "JH"}
_NEAR = {frozenset(pair) for pair in (("AH", "IH"), ("AH", "ER"), ("AA", "AO"), ("IH", "IY"), ("UW", "UH"))}


def read_cmudict(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            word, *phones = line.split()
            entries.setdefault(word.split("(")[0], []).append(phones)
    return entries


def from_ipa(ipa):
    """The source's pronunciation string as CMUdict symbols."""
    out, stress, i = [], 0, 0
    while i < len(ipa):
        char, two = ipa[i], ipa[i:i + 2]
        if char == "'":
            stress = 1
        elif char == "ˌ":
            stress = 2
        elif two in _MULTI:
            out.append(_VOWELS[two] + str(stress))
            stress, i = 0, i + 1
        elif char in _VOWELS:
            out.append(_VOWELS[char] + str(stress))
            stress = 0
        elif char in _CONSONANTS:
            out.append(_CONSONANTS[char])
        i += 1
    return out


def distance(a, b):
    """Edit distance on phonemes; a reduced vowel for another costs little."""
    plain = lambda p: p.rstrip("012")
    table = [[float(i + j) if not i or not j else 0.0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            x, y = plain(a[i - 1]), plain(b[j - 1])
            cost = 0.0 if x == y else 0.3 if frozenset((x, y)) in _NEAR else 1.0
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
    return table[-1][-1]


def primary(phones):
    vowels = [p for p in phones if p[-1] in "012"]
    return next((k for k, p in enumerate(vowels) if p[-1] == "1"), -1)


def role_of(sense):
    parts = set(sense.split("-"))
    return "n" if parts <= {"nou", "adj"} else "v" if parts == {"vrb"} else "o"


def read_senses(data, cmudict, aligner):
    """word -> [{"id", "role", "phones", "source"}] in the order of wordids.tsv."""
    fixes = {(columns[0], columns[1]): columns[2] for _, columns in tables.rows("homograph_fixes.txt")}
    senses = defaultdict(list)
    for row in csv.DictReader((data / "wordids.tsv").open(encoding="utf-8"), delimiter="\t"):
        word, wordid = row["homograph"], row["wordid"]
        sense = wordid[len(word) + 1:] if wordid.startswith(word + "_") else wordid
        if (word, sense) in fixes:
            symbols, source = fixes[(word, sense)].split(), "fix"
        else:
            wanted = from_ipa(row["pronunciation"])
            scored = [(distance(wanted, variant) + (2 if primary(wanted) != primary(variant) else 0), variant)
                      for variant in cmudict.get(word, [])]
            best = min(scored, key=lambda item: item[0]) if scored else None
            symbols, source = (best[1], "cmudict") if best and best[0] <= 1.0 else (wanted, "source")
        if convert.is_arpabet(symbols):
            phones = convert.convert(word, symbols, aligner.align(word, symbols))
        else:
            phones = convert.from_text(" ".join(symbols))
        senses[word].append({"id": sense, "wordid": wordid, "role": role_of(sense), "phones": convert.to_text(phones), "source": source})
    return dict(senses)


def read_examples(directory, senses):
    """word -> [((tokens, position of the homograph), sense index)]."""
    examples = defaultdict(list)
    for path in sorted(directory.glob("*.tsv")):
        for row in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"):
            word = row["homograph"]
            index = next((k for k, sense in enumerate(senses[word]) if sense["wordid"] == row["wordid"]), None)
            if index is None:
                continue
            sentence = row["sentence"]
            start = len(sentence.encode("utf-8")[:int(row["start"])].decode("utf-8", errors="ignore"))
            spans = [(match.start(), match.end(), match.group()) for match in TOKEN.finditer(sentence)]
            position = next((k for k, (a, b, _) in enumerate(spans) if a <= start < b), None)
            if position is None:
                continue
            examples[word].append((([t for _, _, t in spans], position), index))
    return examples


def fit(rows, labels, strength, balanced):
    """Sparse logistic regression. ``rows`` are lists of feature names."""
    names = sorted({name for row in rows for name in row})
    column = {name: k for k, name in enumerate(names)}
    indices = [column[name] for row in rows for name in row]
    pointers = np.cumsum([0] + [len(row) for row in rows])
    matrix = csr_matrix((np.ones(len(indices), dtype=np.float32), indices, pointers), shape=(len(rows), len(names)))
    model = LogisticRegression(penalty="l1", solver="liblinear", C=strength, class_weight="balanced" if balanced else None,
                               max_iter=2000, random_state=0)
    model.fit(matrix, labels)
    weights = {name: round(float(w), 4) for name, w in zip(names, model.coef_[0]) if abs(w) > 1e-6}
    return round(float(model.intercept_[0]), 4), weights


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="WikipediaHomographData/data")
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--built", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0, help="Inverse L1 penalty of the shared noun-or-verb model")
    parser.add_argument("--word-strength", type=float, default=1.0, help="Inverse L1 penalty of the per-word models")
    parser.add_argument("--balanced", action="store_true", help="Weight noun and verb readings equally in training")
    args = parser.parse_args()

    aligner = align.Aligner(json.loads((args.built / "align.json").read_text(encoding="utf-8")))
    senses = read_senses(args.data, read_cmudict(args.cmudict), aligner)
    train = {word: [(features(*sentence_of(*where)), index) for where, index in examples]
             for word, examples in read_examples(args.data / "train", senses).items()}
    train = defaultdict(list, train)
    held_out = read_examples(args.data / "eval", senses)

    def noun_verb(word):
        return sorted(sense["role"] for sense in senses[word]) == ["n", "v"]

    # One model for all noun-or-verb homographs: shared grammatical features, per-word features and prior.
    rows, labels = [], []
    for word in senses:
        if noun_verb(word):
            for names, index in train[word]:
                rows.append(["S|" + name for name in shared(names)] + [f"W|{word}|{name}" for name in names] + [f"W|{word}|prior"])
                labels.append(int(senses[word][index]["role"] == "v"))
    bias, weights = fit(rows, labels, args.strength, args.balanced)
    pooled = {"bias": bias, "weights": {name[2:]: w for name, w in weights.items() if name.startswith("S|")}}
    words = {}
    for word, word_senses in senses.items():
        entry = {"senses": [{key: sense[key] for key in ("id", "role", "phones")} for sense in word_senses]}
        counts = Counter(index for _, index in train[word])
        if noun_verb(word):
            prefix = f"W|{word}|"
            own = {name[len(prefix):]: w for name, w in weights.items() if name.startswith(prefix)}
            entry["model"] = {"kind": "noun-verb", "bias": own.pop("prior", 0.0), "weights": own}
        else:
            biases, sets = [], []
            for index in range(len(word_senses)):
                targets = [int(label == index) for _, label in train[word]]
                if len(word_senses) == 2 and index == 0:
                    biases.append(0.0)                      # two senses: one model, for the second
                    sets.append({})
                elif not counts[index]:
                    biases.append(-1e6)                     # a sense without a single example cannot be chosen
                    sets.append({})
                elif counts[index] == len(targets):
                    biases.append(1e6)
                    sets.append({})
                else:
                    b, w = fit([names for names, _ in train[word]], targets, args.word_strength, False)
                    biases.append(b)
                    sets.append(w)
            entry["model"] = {"kind": "senses", "bias": biases, "weights": sets}
        words[word] = entry

    # Regular -s forms of noun-or-verb homographs ("records", "tears") take the reading of their base,
    # unless the two readings differ only in the last consonant ("use") or the form is listed itself.
    plural_bases = []
    for word, word_senses in senses.items():
        if noun_verb(word) and word + "s" not in senses and word[-1] not in "sxz":
            a, b = (sense["phones"].split() for sense in word_senses)
            if not (a[:-1] == b[:-1] and {a[-1], b[-1]} <= {"s", "z", "θ", "ð"}):
                plural_bases.append(word)
    result = {"words": words, "pooled": pooled, "plural_bases": sorted(plural_bases)}
    path = args.built / "homographs.json"
    path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    from dizione.homographs import Homographs      # the run-time reader scores the held-out sentences
    reader = Homographs(path)

    def predict(word, where):
        chosen = reader.choose(word, *where)
        return next(k for k, sense in enumerate(words[word]["senses"]) if sense["id"] == chosen["id"])

    report = {"homographs": len(senses), "senses": sum(len(v) for v in senses.values()),
              "train_sentences": sum(len(v) for v in train.values()), "eval_sentences": sum(len(v) for v in held_out.values()),
              "weights": len(pooled["weights"]) + sum(len(e["model"]["weights"]) if e["model"]["kind"] == "noun-verb"
                                                      else sum(len(w) for w in e["model"]["weights"]) for e in words.values()),
              "pronunciation_source": dict(Counter(s["source"] for v in senses.values() for s in v)), "balanced": args.balanced}
    for name, group in (("all", list(senses)), ("noun_or_verb", [w for w in senses if noun_verb(w)]),
                        ("other", [w for w in senses if not noun_verb(w)])):
        right = total = majority = 0
        per_word = []
        for word in group:
            common = Counter(index for _, index in train[word]).most_common(1)
            guesses = [(predict(word, names), index) for names, index in held_out[word]]
            if not guesses:
                continue
            right += sum(g == index for g, index in guesses)
            majority += sum(index == (common[0][0] if common else 0) for _, index in guesses)
            total += len(guesses)
            per_word.append(sum(g == index for g, index in guesses) / len(guesses))
        report[name] = {"words": len(per_word), "sentences": total, "accuracy": round(right / total, 4),
                        "accuracy_per_word_mean": round(float(np.mean(per_word)), 4), "always_most_common_sense": round(majority / total, 4)}
    mistakes = Counter()
    for word in senses:
        for names, index in held_out[word]:
            if predict(word, names) != index:
                mistakes[word] += 1
    report["most_mistakes"] = mistakes.most_common(25)
    (args.built / "homographs.report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
