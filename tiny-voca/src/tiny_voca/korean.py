"""Code-mixed Korean + English front end: ``phonemize_mixed(text)``.

"English stays English; Korean stays Korean."

* An English span is passed, as one contiguous phrase, to the original
  ``dizione.phonemize``, so its ids are exactly the ids of the ``own`` front
  end (indices 0..75) and its lexical stress survives.
* A Korean span is pronounced by ``g2pk`` (liaison, nasalisation,
  tensification, ㅎ-deletion, ...), decomposed to Unicode NFD Jamo, and the
  silent initial ㅇ (U+110B) is dropped.  Every Jamo, space and punctuation
  mark becomes one token of ``vocab.UNIFIED_VOCAB`` with ``stress = 0``.
* ``dizione.phonemize`` wraps its output in ``<bos>``/``<eos>``; those are
  stripped from every chunk and applied once around the whole utterance.
* A Korean particle glued to an English word ("Python을") gets no word
  boundary between them: the particle's Jamo follow the English phonemes
  directly.

The result also records ``units``: the alignable groups of the utterance.  One
Korean syllable block is one unit (its Jamo tokens, and the syllable as the
acoustic aligner knows it); one English phrase is one unit with no label,
because the Korean recogniser has no English labels.
"""

from __future__ import annotations

import re
import threading
import unicodedata

from .vocab import (BOS, EOS, PAD, PHONES, PUNCTUATION, SP, UNK, SILENT_INITIAL,
                    KOREAN_PHONES, UNIFIED_TO_ID, UNIFIED_VOCAB)

KOREAN_SET = frozenset(KOREAN_PHONES)
_TOKEN_TO_ID = UNIFIED_TO_ID

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})

# An English span starts with a Latin letter and keeps going over the characters
# that appear inside English words, and over the spaces and punctuation that sit
# between them, so "Hello, world!" is one phrase, exactly as dizione sees it.
# A number only continues a span ("paid $12.99 on 3/14"), so a bare "3시" stays Korean.
_ATOM = r"[A-Za-z][A-Za-z0-9'’\-:/]*"
_CONTINUE = r"(?:[A-Za-z][A-Za-z0-9'’\-:/]*|[$#£€¥]?\d[\d,.:/%-]*)"
_GAP = r"(?:[ \t]*[,.?;:!?][ \t]*|[ \t]+)"
EN_SPAN = re.compile(_ATOM + "(?:" + _GAP + _CONTINUE + ")*")

_state = threading.local()


def g2p():
    """One ``g2pk`` analyser per thread (MeCab's tokenizer is not thread-safe)."""
    analyser = getattr(_state, "g2p", None)
    if analyser is None:
        from g2pk import G2p
        analyser = _state.g2p = G2p()
    return analyser


def dizione():
    phonemizer = getattr(_state, "dizione", None)
    if phonemizer is None:
        import dizione
        phonemizer = _state.dizione = dizione
    return phonemizer


_DIGITS = "영 일 이 삼 사 오 육 칠 팔 구"
_SINO = _DIGITS.split()
_UNIT = ("", "십", "백", "천")
_GROUP = ("", "만", "억", "조")
# Hours are counted in native Korean: 3시 is 세시, not 삼시.
_NATIVE = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
_TENS = ("", "열", "스물")


def _native_korean(value):
    """``1..100`` read the native way, which is what 시 (the hour) takes."""
    if not 1 <= value <= 100:
        return None
    if value == 100:
        return "백"
    tens, ones = divmod(value, 10)
    if value < 10:
        return _NATIVE[value]
    if value == 10:
        return _TENS[1]
    return _TENS[tens] + (_NATIVE[ones] if ones else "")



def _four_digits(value):
    """0..9999 read as Sino-Korean, without a leading '일'."""
    digits = str(value).zfill(4)
    out = ""
    for index, char in enumerate(digits):
        digit = int(char)
        if not digit:
            continue
        power = len(digits) - 1 - index                     # most significant digit first
        unit = _UNIT[power]
        out += (unit if digit == 1 and power else _SINO[digit] + unit)
    return out


def _sino_korean(digits):
    """A run of at most 12 digits, read as one number."""
    text = ""
    for power, group in enumerate(reversed([digits[i:i + 4] for i in range(0, len(digits), 4)])):
        value = int(group)
        if value:
            text = _four_digits(value) + _GROUP[power] + text
    return text or _SINO[0]


def expand_numbers(text):
    """Read leftover digits the way a Korean speaker reads them.

    KSS transcripts already have their numbers written out, so this only runs
    on the digits that survive in free-form prompts (``2026년`` -> ``이천이십육년``).
    Decimals are read first, so ``3.5`` does not become ``삼`` + ``점오``.
    """
    text = re.sub(r"(\d+)\.(\d+)",
                  lambda m: _sino_korean(m.group(1)) + "점" + " ".join(_SINO[int(d)] for d in m.group(2)),
                  text)

    def group(match):
        digits, tail = match.group(1), match.group(2) or ""
        if len(digits) > 12:
            return " ".join(_SINO[int(d)] for d in digits) + tail
        if tail.strip().startswith("시") and len(digits) <= 3:      # the hour is counted natively
            native = _native_korean(int(digits))
            if native:
                return native + tail
        return _sino_korean(digits) + tail

    return re.sub(r"(\d{1,12})(\s*(?:년|월|일|시|분|초|원|개|명|살|세|번|층))?", group, text)


def normalize_ko(text):
    """Safe, conservative normalization: quotes, dashes, whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def segment(text):
    """Yield ``(kind, chunk, gap_before)``: English spans and everything else.

    ``gap_before`` says the chunk starts after whitespace in the source, which
    is where a word boundary token belongs.
    """
    position = 0
    for match in EN_SPAN.finditer(text):
        if match.start() > position:
            yield "ko", text[position:match.start()], False
        yield "en", match.group(), match.start() > 0 and text[match.start() - 1].isspace()
        position = match.end()
    if position < len(text):
        yield "ko", text[position:], False


def _is_syllable(char):
    return 0xAC00 <= ord(char) <= 0xD7A3


def korean_pronounce(text):
    """``g2pk`` output, as pronounced syllable blocks."""
    pronounced = g2p()(text)
    if any(char.isdigit() for char in pronounced):        # g2pk left digits behind
        pronounced = g2p()(expand_numbers(text))
    return pronounced


def korean_tokens(text):
    """``(items, units)`` for one Korean span: Jamo tokens and syllable groups."""
    items, units = [], []
    for char in korean_pronounce(text):
        if char == " ":
            if items and items[-1][0] != SP:
                items.append((SP, 0))
        elif _is_syllable(char):
            start = len(items)
            for jamo in unicodedata.normalize("NFD", char):
                if jamo != SILENT_INITIAL:
                    items.append((jamo, 0))
            if len(items) > start:
                units.append([char, start, len(items)])
        else:
            for mark in unicodedata.normalize("NFD", char):
                items.append((mark if mark in _TOKEN_TO_ID else UNK, 0))
    return items, units


def english_tokens(text):
    """``(items, units, unknown)`` for one English span, via dizione."""
    result = dizione().phonemize(text)
    items = list(zip(result.symbols[1:-1], result.stress[1:-1]))       # strip <bos>/<eos>
    units = [[None, 0, len(items)]] if items else []
    return items, units, list(result.unknown)


def phonemize_mixed(text):
    """Text -> ``Phonemized`` over ``vocab.UNIFIED_VOCAB`` (ids, stress, units)."""
    from .text import Phonemized
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    symbols, stress, units, unknown = [BOS], [0], [], []
    for kind, chunk, gap in segment(normalize_ko(text)):
        if kind == "en":
            items, chunk_units, missing = english_tokens(chunk)
            unknown.extend(missing)
        else:
            items, chunk_units = korean_tokens(chunk)
        if not items:
            continue
        if gap and items[0][0] != SP and items[0][0] not in PUNCTUATION \
                and (symbols[-1] in KOREAN_SET or symbols[-1] in PHONES or symbols[-1] == UNK):
            symbols.append(SP)
            stress.append(0)
            offset = 1
        else:
            offset = 0
        base = len(symbols)
        for symbol, level in items:
            if symbol == SP and symbols[-1] == SP:
                continue
            symbols.append(symbol)
            stress.append(level)
        units.extend([label, start + base - offset, end + base - offset] for label, start, end in chunk_units)
    symbols.append(EOS)
    stress.append(0)
    if not any(symbol in KOREAN_SET or symbol in PHONES for symbol in symbols):
        raise ValueError("the text has nothing to pronounce")
    ids = tuple(_TOKEN_TO_ID[symbol] for symbol in symbols)
    return Phonemized(tuple(symbols), ids, tuple(stress), tuple(unknown), tuple(tuple(unit) for unit in units))
