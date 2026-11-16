"""Pure-Python MeCab stand-in for g2pK, so the front end runs in Pyodide.

g2pK's ``G2p`` calls ``mecab.MeCab().pos(text)`` for morphological analysis.
``python-mecab-ko`` is a native extension with no WASM build, so the browser demo
ships this module as ``mecab.py`` on ``sys.path``.

g2pK only reads the **first letter** of each tag (``utils.annotate`` does
``tag = tag.split("+")[-1]; tag = tag[0]``) and puts it on the last character of
the token, so what matters is the coarse class: N noun, P stem, V verb, A
adjective, E ending, J particle, M modifier, S punctuation, F foreign, B bound
noun. This module segments an eojeol into ``stem + particle`` or
``stem + ending`` with a table of the frequent particles and sentence endings,
and leaves everything else as one noun.

    from mecab import MeCab
    MeCab().pos("합성에")        # [("합성", "NNG"), ("에", "JKB")]

``scripts/check_shim.py`` measures how far this is from the real MeCab.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Particles, longest first: the tag letter that reaches g2pK is the first one.
JOSA = (
    "에게서", "에서부터", "부터까지", "이랑", "으로서", "로서", "으로써", "로서",
    "에서", "에게", "한테", "께로", "으로", "보다", "처럼", "만큼", "대해",
    "이라는", "라는", "라고", "이라고", "이란", "이랑", "이나", "라도", "까지",
    "부터", "마다", "씩", "께", "에", "와", "과", "도", "만", "를", "을", "이",
    "가", "은", "는", "의", "로", "야", "랑", "간", "당", "중", "별", "씩",
)

# Sentence-final and connective endings, longest first.
ENDINGS = (
    "습니다", "ㅂ니다", "입니다", "합니까", "ㅂ니까", "습니다", "셨습니까",
    "이에요", "예요", "세요", "세요", "죠", "다", "요", "고", "며", "면서",
    "면", "서", "니", "니까", "자", "게", "게요", "도록", "려고", "아서", "어서",
    "았어", "었어", "했습니다", "했어요", "한다", "는다", "ㄴ다", "ㄹ다", "던",
    "던데", "더라", "네", "네요", "군", "군요", "거든", "잖아", "잖습니까",
    "읍시다", "ㅂ시다", "라", "라고", "로", "은", "는", "ㄴ", "을", "ㄹ", "기",
    "기요", "침", "함", "게", "케", "야", "야지", "지", "지다", "듯", "듯이",
)

# Adjective stems are tagged VA so the /A class survives; a small list of the
# most frequent ones is enough to make the tag right where it is used.
ADJECTIVES = frozenset("""
좋다 많다 적다 크다 작다 길다 짧다 높다 낮다 넓다 좁다 좋다 새롭다 어렵다 쉽다
예쁘다 아름답다 맛있다 맛없다 재미있다 중요하다 가능다 필요다 비슷다 다름다
맑다 덥다 춥다 따뜻하다 시원하다 빠르다 느리다 강다 약다 정다 낫다 낡다
""".split())

# Bound nouns (의존명사): g2pK sees them as their own token, tagged NNBC → "B".
BOUND = (
    "거야", "거란", "거를", "거", "수", "때", "곳", "것", "줄", "셈", "뿐",
    "점", "대로", "만큼", "차", "중", "적", "여부", "틈", "나름", "비율",
    "경우", "이후", "이전", "동안", "통해", "관련", "대해", "듯", "거동",
)

# Derivational suffixes: XSN → "X".
SUFFIXES = ("적으로", "적인", "적", "화", "성", "률", "율", "권", "용", "용")

_PUNCT = {
    ".": "SF", "!": "SF", "?": "SF", "…": "SF", "~": "SF",
    ",": "SP", ";": "SP", ":": "SP", "·": "SP",
}

_TOKEN = re.compile(r"\s+|[^\s]+")

# Peeling order at each step: a particle, then a sentence ending, then a bound
# noun, then a derivational suffix. Each peel is one g2pK token.
_LAYERS = (
    (JOSA, "JKB"),
    (ENDINGS, "EC"),
    (BOUND, "NNBC"),
    (SUFFIXES, "XSN"),
)


def _is_hangul(text: str) -> bool:
    return any("가" <= char <= "힣" for char in text)


def _peel(word: str, forms) -> tuple[str, str] | None:
    """Split ``word`` into (stem, form) when it ends with one of ``forms``.

    A one-syllable stem is allowed: 하, 가, 보, 없, 할 are all common.
    """
    for form in forms:
        if word.endswith(form) and len(word) - len(form) >= 1:
            return word[: len(word) - len(form)], form
    return None


def _tag_eojeol(word: str) -> list[tuple[str, str]]:
    """Split one eojeol into morphemes with the coarse tags g2pK reads."""
    if not _is_hangul(word):
        return [(word, "SL" if re.search(r"[A-Za-z0-9]", word) else "SF")]
    peels: list[tuple[str, str]] = []
    for forms, tag in _LAYERS:
        peeled = _peel(word, forms)
        if peeled:
            word, form = peeled
            peels.append((form, tag))
            if tag == "EC":                     # a sentence ending closes the eojeol
                break
    if not peels:
        return [(word, "NNG")]
    stem_tag = "VV" if peels[-1][1] == "EC" else "NNG"
    if peels[-1][1] == "EC" and stem_tag == "VV" and (word + "다") in ADJECTIVES:
        stem_tag = "VA"
    out = [(word, stem_tag)]
    out.extend(reversed(peels))
    return out


class MeCab:
    """The ``python-mecab-ko`` surface g2pK uses: ``.pos(text)``.

    ``dictionary`` maps an eojeol to ``[[morpheme, tag], ...]`` as the real MeCab
    analysed it in the training corpus (``scripts/build_mecab_dict.py``). Words
    the corpus contains are analysed exactly; new ones fall back to the tables.
    """

    def __init__(self, dicpath: str = "", dictionary: dict | None = None) -> None:
        self.dicpath = dicpath
        if dictionary is None:
            path = os.environ.get("MECAB_DICT") or Path(__file__).with_name("mecab-dict.json")
            dictionary = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
        self.dictionary = dictionary

    def _tag_eojeol(self, word: str) -> list[tuple[str, str]]:
        known = self.dictionary.get(word)
        if known is not None:
            return [(morph, tag) for morph, tag in known]
        return _tag_eojeol(word)

    def pos(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for piece in _TOKEN.finditer(text):
            token = piece.group()
            if token.isspace():
                continue
            start = 0
            for index, char in enumerate(token):
                if char in _PUNCT:
                    if index > start:
                        out.extend(self._tag_eojeol(token[start:index]))
                    out.append((char, _PUNCT[char]))
                    start = index + 1
            if start < len(token):
                out.extend(self._tag_eojeol(token[start:]))
        return out

    # The rest of the API, in case a caller reaches for it.
    def parse(self, text: str) -> str:
        return "\t".join(f"{token}/{tag}" for token, tag in self.pos(text))

    def nbest(self, text: str, n: int = 1) -> list[dict]:
        return [{"feature": "", "surface": token, "tag": tag} for token, tag in self.pos(text)][:n]

    def yomi(self, text: str) -> str:
        return text


__all__ = ["MeCab"]
