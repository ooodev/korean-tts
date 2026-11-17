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


# ---------------------------------------------------------------------------
# Korean text normalization (TN) for Mode A (``ko-loan``).
#
# Runs before ``segment()``, so digits and symbols never reach the English
# span detector or the acoustic aligner.  The reading table is locked in
# V2_SPEC.md §2.2.  The ``ko`` (Mode C) and ``own`` (Mode B) front ends are
# byte-unchanged: they keep using ``expand_numbers`` as their digit fallback.
# ---------------------------------------------------------------------------

# Native counters: g2pK ``BOUND_NOUNS`` plus the everyday counters from the
# locked reading table (명 장 팩 잔 끼 칸).
_NATIVE_COUNTERS = {
    "군데", "권", "개", "그루", "닥", "대", "두", "마리", "모", "모금", "뭇",
    "발", "발짝", "방", "번", "벌", "보루", "살", "수", "술", "시", "쌈",
    "움큼", "정", "짝", "채", "척", "첩", "축", "켤레", "톨", "통",
    "명", "장", "팩", "잔", "끼", "칸",
}

# Sino counters: read with the Sino-Korean numeral.
_SINO_COUNTERS = {
    "층", "년", "월", "일", "분", "초", "원", "회", "등", "점", "호", "기",
    "차", "페이지", "코어", "세", "박", "세기",
}

_MONTH_NAMES = {
    1: "일", 2: "이", 3: "삼", 4: "사", 5: "오", 6: "유", 7: "칠", 8: "팔",
    9: "구", 10: "시", 11: "십일", 12: "십이",
}

_DIGIT_NAMES_KONG = "공 일 이 삼 사 오 육 칠 팔 구".split()   # digit runs, 0 -> 공
_DIGIT_NAMES_YEONG = "영 일 이 삼 사 오 육 칠 팔 구".split()  # decimals, 0 -> 영

_CURRENCY_UNITS = {"$": "달러", "₩": "원", "€": "유로", "£": "파운드", "¥": "엔"}

_UNIT_NAMES = {
    "km/h": "킬로미터 퍼 시간", "m/s": "미터 퍼 세컨드",
    "km": "킬로미터", "kg": "킬로그램", "mg": "밀리그램", "ml": "밀리리터",
    "cm": "센티미터", "mm": "밀리미터", "°C": "도", "℃": "도",
    "m": "미터", "g": "그램", "l": "리터", "°": "도",
}

# Spoken names for symbols that survive the number rules.  Deliberately excludes
# ``-`` ``'`` ``.`` ``/`` which can sit inside English words before segmentation.
_SYMBOL_NAMES = {
    "+": "플러스", "#": "샵", "@": "앳", "&": "앤", "%": "퍼센트",
    "=": "이퀄", "*": "에스터리스크", "~": "에서",
    "$": "달러", "€": "유로", "£": "파운드", "¥": "엔", "₩": "원", "°": "도",
}

# etc. dictionary: fixed multi-symbol expressions (MeloTTS pattern).
_ETC_PATTERNS = [(re.compile(r"1\+1"), "원플러스원")]

_COUNTER_ALT = "|".join(
    sorted(set(_NATIVE_COUNTERS) | _SINO_COUNTERS, key=len, reverse=True))
_UNIT_ALT = "|".join(
    re.escape(u) for u in sorted(_UNIT_NAMES, key=len, reverse=True))


def _digits_kong(digits):
    """A digit run read digit-by-digit, 0 -> 공 (phones, codes, room numbers)."""
    return "".join(_DIGIT_NAMES_KONG[int(d)] for d in digits)


def _digits_yeong(digits):
    """A decimal fraction read digit-by-digit, 0 -> 영."""
    return "".join(_DIGIT_NAMES_YEONG[int(d)] for d in digits)


def _number_word(n):
    """A non-negative integer read the default (Sino) way; ``0`` -> 영."""
    return _sino_korean(str(int(n)))


def _hour_word(h):
    """The hour reading: ``0`` -> ``영 시``, 1..12 native, 13..24 Sino."""
    if h == 0:
        return "영 시"
    if h <= 12:
        return _native_korean(h) + "시"
    return _number_word(h) + "시"


def _month_word(m):
    name = _MONTH_NAMES.get(m)
    return (name if name is not None else _number_word(m)) + "월"


def _read_counter(num, counter):
    """``num + counter`` -> reading, per the locked counter rules."""
    n = int(num)
    if counter == "살":
        if n == 20:
            return "스무 살"
        if 1 <= n <= 20:
            return _native_korean(n) + "살"
        return _number_word(n) + " 살"
    if counter in ("세", "박"):                       # age / nights of stay
        return _number_word(n) + " " + counter
    if counter in ("번", "호") and len(num) >= 3:      # codes: digit-by-digit
        return _digits_kong(num) + counter
    if counter in _NATIVE_COUNTERS:
        if 1 <= n <= 20:
            return _native_korean(n) + counter
        return _number_word(n) + counter
    return _number_word(n) + counter                  # Sino counter


def korean_normalize(text):
    """Full Korean TN for Mode A; see the locked table in V2_SPEC.md §2.2."""

    # etc. dictionary: fixed multi-symbol expressions.
    for pattern, replacement in _ETC_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3-part date: 2026-03-14 / 2026/3/14.
    text = re.sub(
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        lambda m: _number_word(int(m.group(1))) + "년 "
        + _month_word(int(m.group(2))) + " "
        + _number_word(int(m.group(3))) + "일", text)

    # currency: $12.99 / $12 / ₩3000 / €20 / £5 / ¥100.
    def _currency(m):
        out = _number_word(int(m.group(2).replace(",", ""))) + " " \
            + _CURRENCY_UNITS[m.group(1)]
        if m.group(3):
            out += " " + _number_word(int(m.group(3))) + " 센트"
        return out
    text = re.sub(r"([$₩€£¥])(\d+(?:,\d{3})*)(?:\.(\d{1,2}))?", _currency, text)

    # percent: 5% / 3.5%.
    def _percent(m):
        num = m.group(1)
        if "." in num:
            whole, frac = num.split(".", 1)
            return _number_word(int(whole)) + "점" + _digits_yeong(frac) + " 퍼센트"
        return _number_word(int(num)) + " 퍼센트"
    text = re.sub(r"(\d+(?:\.\d+)?)%", _percent, text)

    # time of day: 14:30 / 13:15 / 20:00 / 14:30:22.
    def _time(m):
        out = _hour_word(int(m.group(1)))
        minute = int(m.group(2))
        if minute:
            out += " " + _number_word(minute) + "분"
        second = m.group(3)
        if second and int(second):
            out += " " + _number_word(int(second)) + "초"
        return out
    text = re.sub(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", _time, text)

    # phone number: 010-1234-5678 / 01012345678 (groups of 3-4-4).
    def _phone(m):
        return (_digits_kong(m.group(1)) + " " + _digits_kong(m.group(2))
                + " " + _digits_kong(m.group(3)))
    text = re.sub(r"(01[01])(?:-?)?(\d{4})(?:-?)?(\d{4})", _phone, text)

    # range: 100~200명 -> 백에서 이백 명.
    def _range(m):
        out = _number_word(int(m.group(1))) + "에서 " + _number_word(int(m.group(2)))
        if m.group(3):
            out += " " + m.group(3).strip()
        return out
    text = re.sub(r"(\d+)\s*~\s*(\d+)(\s*(?:" + _COUNTER_ALT + "))?", _range, text)

    # unit suffix: 10km / 2kg / 500ml / 60°C.
    def _unit(m):
        num = m.group(1)
        if "." in num:
            whole, frac = num.split(".", 1)
            num_reading = _number_word(int(whole)) + "점" + _digits_yeong(frac)
        else:
            num_reading = _number_word(int(num))
        return num_reading + " " + _UNIT_NAMES[m.group(2)]
    # The unit must not be followed by an ASCII letter/digit, but may be followed
    # by a Korean syllable ("10km를"): ``\b`` treats Hangul as word characters and
    # would refuse to match there, so use an explicit ASCII-only lookahead.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(" + _UNIT_ALT + r")" + r"(?![A-Za-z0-9])", _unit, text)

    # decimals: 3.5 / 3.05.
    text = re.sub(
        r"(\d+)\.(\d+)",
        lambda m: _number_word(int(m.group(1))) + "점" + _digits_yeong(m.group(2)),
        text)

    # a/b: date if M <= 12 and D <= 31, else a fraction (삼 분의 십사).
    def _slash(m):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 12 and b <= 31:
            return _month_word(a) + " " + _number_word(b) + "일"
        return _number_word(b) + " 분의 " + _number_word(a)
    text = re.sub(r"(\d{1,2})/(\d{1,2})", _slash, text)

    # ordinal: 3번째 -> 세 번째.
    def _ordinal(m):
        n = int(m.group(1))
        native = _native_korean(n)
        return (native if native else _number_word(n)) + " 번째"
    text = re.sub(r"(\d+)번째", _ordinal, text)

    # the month: 6월 -> 유월 / 10월 -> 시월 (standalone or in a particle date).
    text = re.sub(r"(\d+)월", lambda m: _month_word(int(m.group(1))), text)

    # the hour: 3시 / 0시 / 14시.
    text = re.sub(r"(\d+)시", lambda m: _hour_word(int(m.group(1))), text)

    # counters: 3개 / 3년 / 3세 / 20살 / 123번.
    text = re.sub(r"(\d+)(" + _COUNTER_ALT + r")",
                  lambda m: _read_counter(m.group(1), m.group(2)), text)

    # code-like X + digits: K2 -> K이.
    text = re.sub(r"([A-Z])(\d+)",
                  lambda m: m.group(1) + _digits_kong(m.group(2)), text)
    # >= 5 consecutive digits -> digit-by-digit.
    text = re.sub(r"\d{5,}", lambda m: _digits_kong(m.group(0)), text)
    # remaining digit runs -> Sino (0 -> 영).
    text = re.sub(r"\d+", lambda m: _number_word(int(m.group(0))), text)

    # remaining symbols -> spoken Korean names.
    for symbol, name in _SYMBOL_NAMES.items():
        text = text.replace(symbol, name)

    return text


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


def loan_tokens(text):
    """``(items, units, unknown)`` for one English span read as Korean loanwords.

    Mode A (``ko-loan``): the span is spelled out as Korean syllables by
    ``loanwords.loanword_span`` (dictionary -> acronym -> IPA fallback), and each
    syllable is decomposed to Jamo with ``stress = 0``.  One syllable is one unit,
    labelled with the Hangul syllable, so ``kalign`` aligns it like a Korean word.
    """
    from .loanwords import loanword_span
    items, units = [], []
    raw_items, _oov = loanword_span(text)   # IPA-fallback syllables are valid, not UNK
    for token, _ in raw_items:
        if token == " ":
            if items and items[-1][0] != SP:
                items.append((SP, 0))
        elif _is_syllable(token):
            start = len(items)
            for jamo in unicodedata.normalize("NFD", token):
                if jamo != SILENT_INITIAL:
                    items.append((jamo, 0))
            if len(items) > start:
                units.append([token, start, len(items)])
        else:
            for mark in unicodedata.normalize("NFD", token):
                items.append((mark if mark in _TOKEN_TO_ID else UNK, 0))
    return items, units, []


def phonemize_mixed(text, frontend=None):
    """Text -> ``Phonemized`` over ``vocab.UNIFIED_VOCAB`` (ids, stress, units).

    ``frontend`` is the mode: ``"ko"`` (Mode C, byte-unchanged) or ``"ko-loan"``
    (Mode A: Korean TN before segmentation, loanword Jamo for English spans).
    ``None`` defers to ``text.frontend()``.
    """
    from .text import Phonemized
    if frontend is None:
        from . import text as _text_mod
        frontend = _text_mod.frontend()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    normalized = normalize_ko(text)
    if frontend == "ko-loan":
        normalized = korean_normalize(normalized)
    symbols, stress, units, unknown = [BOS], [0], [], []
    for kind, chunk, gap in segment(normalized):
        if kind == "en":
            if frontend == "ko-loan":
                items, chunk_units, missing = loan_tokens(chunk)
            else:
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
