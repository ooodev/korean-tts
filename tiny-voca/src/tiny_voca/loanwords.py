"""Mode A loanword front end: English word -> Korean Jamo (loanword orthography).

Pipeline per English span (see V2_SPEC.md §2.3):

1. dictionary lookup (``data/loanwords.tsv``, override with ``TINY_VOCA_LOANWORDS``)
2. acronym / letter spelling (KOREA -> 코리아, otherwise letter by letter)
3. IPA -> Jamo two-phase engine over ``dizione`` IPA (OOV fallback)

The engine is deterministic: phase 1 emits a linear stream of ``(kind, jamo)``
items (onset ``o`` / nucleus ``n`` / coda ``c``) under the 2014 loanword rules,
phase 2 folds that stream into syllables.  Standard spellings live in the
dictionary; the engine keeps an out-of-vocabulary word pronounceable and
alignable.  Anything unmappable is counted, never guessed.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import unicodedata

# --- dictionary ---------------------------------------------------------------

_STATE = threading.local()
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")

# Letter names per CoreaSpeech N2gk+ SINGLE_LETTER_MAPPING / MeloTTS
# english_dictionary (R=알, Z=제트 — the two references disagree on Z; 제트 is
# the 2-of-3 majority).
LETTER_NAMES = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지",
    "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔",
    "O": "오", "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유",
    "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "제트",
}

def _dict_path():
    return os.environ.get(
        "TINY_VOCA_LOANWORDS",
        os.path.join(os.path.dirname(__file__), "data", "loanwords.tsv"))

def load_dictionary(reload=False):
    """Lower-cased English word -> standard Korean loanspelling. Cached."""
    cache = getattr(_STATE, "dict", None)
    if cache is not None and not reload:
        return cache
    entries = {}
    path = _dict_path()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    entries[parts[0].lower()] = parts[1]
    _STATE.dict = entries
    return entries

def dictionary_fingerprint():
    """sha256 of the canonical dictionary bytes, for ``frontend_identity``."""
    path = _dict_path()
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    return "missing"

def _is_acronym(word):
    """All-caps span of 2..6 letters: letter-spellable (PC -> 피씨)."""
    return 2 <= len(word) <= 6 and word.isupper() and word.isalpha()

# --- IPA -> Jamo tables -------------------------------------------------------
# Input alphabet: the 66 phones of vocab.PHONES (dizione's inventory).  Verified
# against dizione IPA output (probe_dizione_g2p.py).

# Onset consonant (2014 plain series).
ONSET = {
    "p": "ㅂ", "b": "ㅂ", "t": "ㄷ", "d": "ㄷ", "k": "ㄱ", "ɡ": "ㄱ",
    "tʃ": "ㅊ", "dʒ": "ㅈ", "f": "ㅍ", "v": "ㅂ", "θ": "ㅅ", "ð": "ㄷ",
    "s": "ㅅ", "z": "ㅈ", "ʃ": "ㅅ", "ʒ": "ㅈ", "h": "ㅎ",
    "m": "ㅁ", "n": "ㄴ", "ŋ": "ㄴ", "l": "ㄹ", "ɹ": "ㄹ", "ɾ": "ㄹ", "j": "ㄹ",
    "w": "ㄹ",                                     # wide -> 와이드 (w not merged)
}

# Syllable nuclei.  Single vowel -> one jungseong.  ɜː -> 으 (thirty -> 서티,
# 퍼스트); ɒ -> ㅗ (chocolate -> 초콜릿, coffee -> 커피).
NUCLEUS = {
    "iː": "ㅣ", "i": "ㅣ", "ɪ": "ㅣ", "æ": "에", "ɛ": "에", "ɜː": "으", "ə": "으",
    "ʌ": "ㅓ", "ɑː": "ㅏ", "ɔ": "ㅗ", "ɔː": "ㅗ", "ʊ": "ㅜ", "uː": "ㅜ", "u": "ㅜ",
    "ɒ": "ㅗ", "eɪ": "에", "oʊ": "ㅗ", "oː": "ㅗ", "ɐ": "아",
}

# Diphthongs that split into two nuclei across a syllable boundary
# (my -> 마이, how -> 하우, fire -> 파이어).
SPLIT = {
    "aɪ": ("ㅏ", "ㅣ"), "aʊ": ("ㅏ", "ㅜ"), "iə": ("ㅕ", "ㅣ"), "ɔɪ": ("ㅗ", "ㅣ"),
}

# w/j glides merge with the following vowel: value = (onset, nucleus, absorb).
# absorb=True: every preceding consonant gets its own 으 (quick -> 퀵, 퀸).
# onset "ㄹ": back rounded vowels take a ㄹ onset (water -> 워터, yard -> 야드,
# yellow -> 옐로우, yes -> 예스).
_W_MERGE = {
    "ɪ": ("", "ㅟ", True), "iː": ("", "ㅟ", True),
    "ɛ": ("", "ㅓ", True), "æ": ("", "ㅘ", True),
    "ʌ": ("", "ㅓ", True), "ə": ("", "ㅓ", True),
    "ɑ": ("", "ㅘ", False), "ɔ": ("", "ㅗ", False),
    "ɑː": ("ㄹ", "ㅝ", False), "ɔː": ("ㄹ", "ㅝ", False), "oʊ": ("", "ㅗ", False),
    "ʊ": ("", "ㅜ", True), "uː": ("", "ㅟ", True),
    "eɪ": ("ㄹ", "ㅔ", False), "aɪ": ("ㄹ", "ㅏ", False), "aʊ": ("ㄹ", "ㅏ", False),
    "ɜː": ("", "ㅝ", True),                            # world -> 월드
}
_J_MERGE = {
    "uː": ("", "ㅜ", False), "ʊ": ("", "ㅟ", True),
    "ɪ": ("", "ㅣ", True), "iː": ("", "ㅣ", True),
    "ʌ": ("ㄹ", "ㅛ", False), "ɛ": ("ㄹ", "ㅔ", False), "æ": ("ㄹ", "ㅔ", False),
    "ɑː": ("ㄹ", "ㅗ", False), "ɔː": ("ㄹ", "ㅗ", False), "ɔ": ("ㄹ", "ㅗ", False),
}

# R-colored nucleus: ɚ = 으 + coda ㄹ (color -> 컬러, shopper -> 쇼퍼);
# after a flap ɾ the nucleus is ㅓ (water -> 워터, butter -> 버터); after aʊ
# it absorbs the ㅜ into ㅝ (power -> 파워); a following coda consonant takes
# the ㄹ slot (vision -> 비전, percent -> 퍼센트, turn -> 턴).
_RHOTIC = "ɚ"

# Coda (batchim) spelling.  The 2014 loanword rule plus standard spellings:
# t -> ㅅ, d -> ㄹ.
CODA_FINAL = {
    "p": "ㅂ", "b": "ㅂ", "t": "ㅅ", "d": "ㄹ", "k": "ㄱ", "ɡ": "ㄱ",
    "m": "ㅁ", "n": "ㄴ", "ŋ": "ㄴ", "l": "ㄹ", "ɹ": "ㄹ", "ɾ": "ㄹ",
}
# Final fricatives open a fresh syllable (fish -> 피시, bus -> 버스,
# month -> 먼스, church -> 처치).
FINAL_OPEN = {
    "s": "으", "z": "으", "ʃ": "ㅣ", "ʒ": "으", "f": "으", "v": "으",
    "θ": "으", "ð": "으", "tʃ": "ㅣ", "dʒ": "ㅣ",
}
# A word-final stop takes its coda only after these plain monophthongs
# (set -> 셋, shop -> 십, dog -> 독, bag -> 백, jeep -> 지프); after
# diphthongs / r-colored vowels it opens 으 (wide -> 와이드, white -> 화이트,
# smart -> 스마트, bird -> 버드).  Nasals/liquids are always coda.
_CODA_OK = frozenset(("iː", "i", "ɪ", "æ", "ɛ", "ə", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ",
                       "uː", "u", "oː", "ɐ", "ɒ"))

# dizione tokens that are really two phones written together (vocab.VOWELS tail).
_CLUSTER_VOWELS = {
    "ɑːɹ": ("ɑː", "ɹ"), "ɔːɹ": ("ɔː", "ɹ"), "oːɹ": ("oː", "ɹ"),
    "ɪɹ": ("ɪ", "ɹ"), "ɛɹ": ("ɛ", "ɹ"), "ʊɹ": ("ʊ", "ɹ"),
    "aɪɚ": ("aɪ", "ɚ"), "aɪə": ("aɪ", "ə"), "əl": ("ə", "l"), "n̩": ("n",),
}

def _expand_phones(symbols):
    """Split cluster tokens into ``[(token, rhotic_coda_flag), ...]``.

    The ɹ of an r-colored cluster vowel (ɑːɹ) is the coda of that vowel, not a
    pending onset."""
    out = []
    for sym in symbols:
        parts = _CLUSTER_VOWELS.get(sym)
        if parts is None:
            out.append((sym, False))
        else:
            out.append((parts[0], False))
            for extra in parts[1:]:
                out.append((extra, extra in ("ɹ", "ɾ")))
    return out

# The tables above are written with Hangul compatibility jamo (U+31xx) and a
# couple of composed syllables (에, 으) for readability.  The syllable math
# needs jamo forms (U+11xx), and the compat block order does NOT match the
# jamo block order (e.g. ㄹ=U+3139 -> U+1105, ㅓ=U+3153 -> U+1165), so every
# mapping is keyed by Unicode character name — verified at import time.

def _build_name_maps():
    def by_name(lo, hi):
        out = {}
        for cp in range(lo, hi + 1):
            try:
                out[unicodedata.name(chr(cp))] = chr(cp)
            except ValueError:
                pass
        return out
    return (by_name(0x1100, 0x1112), by_name(0x1161, 0x1175),
            by_name(0x11A8, 0x11C2))

_CHO_NAME, _JUNG_NAME, _JONG_NAME = _build_name_maps()

def _compat_name(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if not name.startswith("HANGUL LETTER "):
        return None
    return name[len("HANGUL LETTER "):]

def _resolve(ch, prefix, table, where):
    tail = _compat_name(ch)
    if tail is None:
        raise ValueError(f"unmappable {where} jamo: {ch!r}")
    try:
        return table[prefix + tail]
    except KeyError:
        raise ValueError(f"unmappable {where} jamo: {ch!r}") from None

def _cho_of(ch):
    """Choseong jamo form (U+1100-U+1112) of a compat jamo / composed syllable."""
    cp = ord(ch)
    if 0x1100 <= cp <= 0x1112:
        return ch
    if 0xAC00 <= cp <= 0xD7A3:
        return unicodedata.normalize("NFD", ch)[0]
    return _resolve(ch, "HANGUL CHOSEONG ", _CHO_NAME, "cho")

def _jung_of(ch):
    """Jungseong jamo form (U+1161-U+1175) of a compat jamo / composed
    syllable (composite nuclei like ㅝ/ㅟ are single jamo U+116A-U+1175)."""
    cp = ord(ch)
    if 0x1161 <= cp <= 0x1175:
        return ch
    if 0xAC00 <= cp <= 0xD7A3:
        for p in unicodedata.normalize("NFD", ch)[1:]:
            if 0x1161 <= ord(p) <= 0x1175:
                return p
    return _resolve(ch, "HANGUL JUNGSEONG ", _JUNG_NAME, "jung")

def _jong_of(ch):
    """Jongseong jamo form (U+11A8-U+11C2) of a compat jamo."""
    cp = ord(ch)
    if 0x11A8 <= cp <= 0x11C2:
        return ch
    return _resolve(ch, "HANGUL JONGSEONG ", _JONG_NAME, "jong")

def _syllable(onset, nucleus, coda):
    """Compose one Hangul syllable.

    Empty onset -> silent initial U+110B (dropped downstream by
    ``korean_tokens``); empty coda -> no batchim.  Verified formula:
    ``0xAC00 + (cho*21 + jung)*28 + pos`` with ``pos = ord(jong) - 0x11A8 + 1``
    (각=AC01, 갓=AC13, 강=AC15, 을=C744, 권=AD8C)."""
    onset = _cho_of(onset or "\u110b")
    nucleus = _jung_of(nucleus)
    pos = 0 if not coda else ord(_jong_of(coda)) - 0x11A8 + 1
    return chr(0xAC00 + (ord(onset) - 0x1100) * 21 * 28
               + (ord(nucleus) - 0x1161) * 28 + pos)

def _fold(items):
    """(kind, jamo) stream -> composed syllables.  Consecutive nuclei each open
    a new syllable (aɪ -> 마이); a coda attaches to the open syllable."""
    syllables = []
    o = n = c = None
    for kind, jamo in items:
        if kind == "c":
            c = jamo
            continue
        if n is not None:
            syllables.append(_syllable(o, n, c))
            o, n, c = None, None, None
        if kind == "o":
            o = jamo
        else:
            n = jamo
    if n is not None:
        syllables.append(_syllable(o, n, c))
    return syllables

def ipa_to_jamo(symbols):
    """One dizione word: IPA token list -> ``(syllables, unknown_count)``.

    Deterministic OOV fallback; standard spellings live in the dictionary."""
    tokens = _expand_phones(symbols)
    items, unknown = [], 0
    last_nucleus = None       # last emitted nucleus token ("aʊ", "ɑːɹ", ...)
    last_has_coda = False

    def add(kind, jamo):
        items.append((kind, jamo))

    def is_nucleus(tok):
        return tok in NUCLEUS or tok in SPLIT or tok == _RHOTIC

    def onset_jamo(c):
        nonlocal unknown
        j = ONSET.get(c)
        if j is None:
            unknown += 1
            return "\u110b"
        return j

    def emit_cluster(cluster, vowel, nxt):
        """Emit Jamo for consonant cluster before nucleus token ``vowel``;
        ``nxt`` is the token after the nucleus (for ɚ lookahead)."""
        nonlocal last_nucleus, last_has_coda, unknown
        merged = None
        onset_r = ""
        absorb = False
        body = list(cluster)
        if body and body[-1] in ("w", "j") and vowel is not None and vowel != _RHOTIC:
            table = _W_MERGE if body[-1] == "w" else _J_MERGE
            if vowel in table:
                onset_r, merged, absorb = table[vowel]
                body = body[:-1]
        first_onset = None
        shared = False
        if body and merged is None:
            if last_nucleus is None or last_has_coda:
                for c in body[:-1]:
                    add("o", onset_jamo(c)); add("n", "으")
                first_onset = body[-1]
            elif len(body) == 1:
                c = body[0]
                if c in ("l", "ɹ", "ɾ") and vowel == _RHOTIC:
                    add("c", "ㄹ")                     # shared ㄹ (color -> 컬러)
                    shared = True
                else:
                    first_onset = c
            else:
                cj = CODA_FINAL.get(body[0])
                if cj is None:
                    nonlocal unknown
                    unknown += 1
                else:
                    add("c", cj)                       # coda of the open vowel
                last_has_coda = True
                for c in body[1:-1]:
                    add("o", onset_jamo(c)); add("n", "으")
                first_onset = body[-1]
        if merged is not None:
            if absorb:
                for c in body:
                    add("o", onset_jamo(c)); add("n", "으")
            elif first_onset is not None:
                add("o", onset_jamo(first_onset))
            if onset_r:
                add("o", onset_r)
            add("n", merged)
            last_nucleus, last_has_coda = vowel, False
        elif vowel in SPLIT:
            if first_onset is not None:
                add("o", onset_jamo(first_onset))
            first_j, second_j = SPLIT[vowel]
            if vowel == "aʊ" and nxt == _RHOTIC:
                add("n", first_j)                      # ɚ absorbs the ㅜ (파워)
            else:
                add("n", first_j); add("n", second_j)
            last_nucleus, last_has_coda = vowel, False
        elif vowel == _RHOTIC:
            coda_follows = nxt in CODA_FINAL
            if shared:
                add("o", "ㄹ")
            elif first_onset is not None:
                add("o", onset_jamo(first_onset))
            elif coda_follows:
                add("o", "ㄹ")
            if first_onset == "ɾ":
                add("n", "ㅓ")                          # 워터, 버터
            elif last_nucleus == "aʊ":
                add("n", "ㅝ")                          # 파워, 샤워
            else:
                add("n", "으")
            if coda_follows:
                add("c", CODA_FINAL[nxt])
            else:
                add("c", "ㄹ")
            last_nucleus, last_has_coda = vowel, True
        else:
            if first_onset is not None:
                add("o", onset_jamo(first_onset))
            add("n", NUCLEUS[vowel])
            last_nucleus, last_has_coda = vowel, False

    def emit_final(pending):
        """Word-final consonant cluster.  /st/ -> 스트 (two 으), a 2-consonant
        cluster with a non-t coda folds to one 으 (desk -> 데스크, world ->
        월드), t never folds (test -> 테스트)."""
        nonlocal last_nucleus, last_has_coda, unknown
        cluster = [c for c in pending if c not in ("ɹ", "ɾ")]   # final rhotic drops
        if not cluster:
            return
        if len(cluster) == 1:
            c = cluster[0]
            fo = FINAL_OPEN.get(c)
            cj = CODA_FINAL.get(c)
            if fo is not None:
                add("o", onset_jamo(c)); add("n", fo)
            elif c in ("m", "n", "ŋ", "l") or last_nucleus in _CODA_OK:
                add("c", cj)
            else:
                add("o", onset_jamo(c)); add("n", "으"); add("c", cj)
            return
        c_last = cluster[-1]
        cj_l = CODA_FINAL.get(c_last)
        if len(cluster) == 2 and cj_l is not None and c_last != "t":
            add("o", onset_jamo(cluster[0])); add("n", "으"); add("c", cj_l)
        else:
            for c in cluster[:-1]:
                add("o", onset_jamo(c)); add("n", "으")
            fo_l = FINAL_OPEN.get(c_last)
            if fo_l is not None:
                add("o", onset_jamo(c_last)); add("n", fo_l)
            else:
                add("o", onset_jamo(c_last)); add("n", "으")
                cj_l = CODA_FINAL.get(c_last)
                if cj_l is None:
                    unknown += 1
                else:
                    add("c", cj_l)

    i, n = 0, len(tokens)
    pending = []
    while i < n:
        tok, rhotic_coda = tokens[i]
        if rhotic_coda:
            add("c", "ㄹ")
            last_has_coda = True
            if last_nucleus is not None and last_nucleus not in SPLIT:
                last_nucleus = last_nucleus + "ɹ"        # rhotic-closed marker
            i += 1
            continue
        if is_nucleus(tok):
            nxt = tokens[i + 1][0] if i + 1 < n else None
            emit_cluster(pending, tok, nxt)
            pending = []
            if tok == _RHOTIC and nxt in CODA_FINAL:
                i += 1                                  # consume lookahead coda
        elif tok == "ʔ":
            pass                                        # glottal stop: drop
        elif tok == "h" and i == n - 1:
            pass                                        # word-final h: drop
        elif tok in ONSET or tok in CODA_FINAL or tok in ("j", "w"):
            pending.append(tok)
        # else: punctuation / unlisted token — skip (spans are word-filtered)
        i += 1
    if pending:
        emit_final(pending)
    return _fold(items), unknown

# --- word / span dispatch -----------------------------------------------------

def word_to_syllables(word, dictionary):
    """One English word -> Korean syllables, or ``None`` if unconvertible."""
    key = word.lower()
    if key in dictionary:
        return list(dictionary[key])
    if _is_acronym(word):
        return [ch for ch in "".join(LETTER_NAMES[c] for c in word)
                if 0xAC00 <= ord(ch) <= 0xD7A3]
    return None

def loanword_tokens(word, dictionary=None):
    """Dictionary / acronym / IPA-engine fallback for one English word.

    Returns ``(syllable_list, unknown_count)``."""
    dictionary = load_dictionary() if dictionary is None else dictionary
    result = word_to_syllables(word, dictionary)
    if result is not None:
        return result, 0
    phonemizer = getattr(_STATE, "dizione", None)
    if phonemizer is None:
        import dizione
        phonemizer = _STATE.dizione = dizione
    result = phonemizer.phonemize(word)
    symbols = list(result.symbols[1:-1])                # strip <bos>/<eos>
    if not symbols:
        return None
    return ipa_to_jamo(symbols)

def loanword_span(span, dictionary=None):
    """One English span -> ``((token, 0), ...)``; words space-joined."""
    dictionary = load_dictionary() if dictionary is None else dictionary
    out, unknown = [], 0
    for word in _WORD_RE.findall(span):
        if not word.strip(" \t,.?;:!?"):
            continue
        result, word_unknown = loanword_tokens(word, dictionary)
        if result is None:
            continue
        unknown += word_unknown
        if out:
            out.append((" ", 0))
        out.extend((s, 0) for s in result)
    return out, unknown
