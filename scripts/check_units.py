"""Invariant checks for the code-mixed front end: units must index the tokens exactly.

    python scripts/check_units.py
"""
import unicodedata

from tiny_voca.korean import SILENT_INITIAL, phonemize_mixed
from tiny_voca.text import PHONE_SET, PUNCTUATION

TEXTS = [
    "안녕하세요. 맑은 날씨가 좋아요!",
    "오늘 Python으로 API 테스트를 했어요.",
    "Python을 실행하세요.",
    "Hello, world!",
    "그는 괜찮은 척하고 있는 것 같았어",
    "2026년 11월 15일 오후 3시 30분입니다.",
    "와, 정말 대단하네요! 드디어 한국어 음성 합성에 성공했습니다!",
    "저는 보통 20분 정도 낮잠을 자고 나면 오후 업무에 훨씬 더 집중할 수 있습니다.",
]


def jamos(syllable):
    return "".join(j for j in unicodedata.normalize("NFD", syllable) if j != SILENT_INITIAL)


def check(text):
    phones = phonemize_mixed(text)
    symbols = list(phones.symbols)
    problems = []
    if len(symbols) != len(phones.ids) or len(symbols) != len(phones.stress):
        problems.append("ids/stress/symbols length mismatch")
    cursor = 0
    for label, start, end in phones.units:
        if not 0 <= start < end <= len(symbols):
            problems.append(f"unit {label!r} range [{start}:{end}] out of bounds")
            continue
        if start < cursor:
            problems.append(f"unit {label!r} overlaps the previous unit")
        cursor = end
        tokens = symbols[start:end]
        # An English unit is one whole phrase, so its commas and periods are inside it;
        # the aligner keeps only the phone tokens of a unit.
        if any(symbol not in PHONE_SET and symbol != "<unk>" and symbol not in PUNCTUATION
               for symbol in tokens):
            problems.append(f"unit {label!r} [{start}:{end}] = {tokens} has unalignable tokens")
        if label is not None and "".join(tokens) != jamos(label):
            problems.append(f"syllable {label!r} [{start}:{end}] = {tokens} != {list(jamos(label))}")
    # every phone token must belong to exactly one unit
    covered = set()
    for _, start, end in phones.units:
        covered.update(range(start, min(end, len(symbols))))
    missing = [i for i, symbol in enumerate(symbols)
               if (symbol in PHONE_SET or symbol == "<unk>") and i not in covered]
    if missing:
        problems.append(f"phones outside any unit: {[(i, symbols[i]) for i in missing]}")
    print(("ok   " if not problems else "FAIL ") + text)
    for problem in problems:
        print("      ", problem)
    return not problems


if __name__ == "__main__":
    raise SystemExit(0 if all(check(text) for text in TEXTS) else 1)
