"""Token inventories shared by the text front end, the model and the aligner.

Two inventories exist.  Indices ``0 .. 75`` are the original ``dizione``
English vocabulary, in its original order, so every id produced by
``dizione.phonemize`` is valid in both.  The Korean front end appends the 66
Unicode NFD Jamo that are not already English symbols, giving 142 tokens.
The order is append-only: checkpoints index the embedding table by position.
"""

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
              "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ")
VOWELS = ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
          "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
          "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩")
PHONES = CONSONANTS + VOWELS
EN_SYMBOLS = SPECIALS + PUNCTUATION + PHONES          # 76 tokens, indices 0..75
EN_VOICELESS = frozenset(("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ"))

# --- Korean NFD Jamo -------------------------------------------------------
# 19 leading consonants minus the silent initial ㅇ (U+110B): 18 tokens.
CHOSEONG = tuple(chr(code) for code in range(0x1100, 0x1113) if code != 0x110B)
JUNGSEONG = tuple(chr(code) for code in range(0x1161, 0x1176))               # 21
JONGSEONG = tuple(chr(code) for code in range(0x11A8, 0x11C3))               # 27 (keeps final ㅇ U+11BC)
SILENT_INITIAL = "\u110b"
KOREAN_PHONES = CHOSEONG + JUNGSEONG + JONGSEONG                              # 66
KOREAN_VOWELS = JUNGSEONG
# The plain (non-tense) obstruents plus the aspirated series; tense consonants
# are phonemically voiceless too.  Used by the label-free voicing probes only.
_KOREAN_OBSTRUENTS = ("ㄱ", "ㄲ", "ㄷ", "ㄸ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
KOREAN_VOICELESS = frozenset(_KOREAN_OBSTRUENTS)

UNIFIED_VOCAB = EN_SYMBOLS + KOREAN_PHONES            # 142 tokens, indices 0..141
UNIFIED_TO_ID = {symbol: index for index, symbol in enumerate(UNIFIED_VOCAB)}
EN_TO_ID = {symbol: index for index, symbol in enumerate(EN_SYMBOLS)}

KOREAN_PHONE_IDS = frozenset(UNIFIED_TO_ID[j] for j in KOREAN_PHONES)
KOREAN_VOWEL_IDS = frozenset(UNIFIED_TO_ID[j] for j in KOREAN_VOWELS)


def inventory(frontend):
    """The token list of one front end."""
    return UNIFIED_VOCAB if frontend == "ko" else EN_SYMBOLS


def phones(frontend):
    """Tokens that must be realised as audio frames."""
    return PHONES + (KOREAN_PHONES if frontend == "ko" else ())


def vowels(frontend):
    return VOWELS + (KOREAN_VOWELS if frontend == "ko" else ())


def voiceless(frontend):
    return EN_VOICELESS | (KOREAN_VOICELESS if frontend == "ko" else frozenset())
