"""The code-mixed Korean (NFD Jamo) + English (dizione) front end.

"English stays English; Korean stays Korean": English spans keep their
original dizione ids and lexical stress, Korean spans become NFD Jamo with
``stress = 0``, and the two share one 142-token inventory whose first 76 ids
are exactly the English ones.

Skipped when ``g2pk`` or dizione's built data is missing.
"""

import os
import unittest
import unicodedata
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")
if "DIZIONE_DATA" not in os.environ:
    for candidate in (Path(__file__).resolve().parents[2] / "weights" / "phonemizer",
                      Path(__file__).resolve().parents[1] / "weights" / "phonemizer"):
        if (candidate / "lexicon.tsv").exists():
            os.environ["DIZIONE_DATA"] = str(candidate)
            break

from tiny_voca import text  # noqa: E402
from tiny_voca.vocab import (CHOSEONG, EN_SYMBOLS, JONGSEONG, JUNGSEONG,  # noqa: E402
                             KOREAN_PHONES, SILENT_INITIAL, UNIFIED_VOCAB)

text.set_frontend("ko")        # a sibling module may have selected eSpeak first


def setUpModule():
    text.set_frontend("ko")


def tearDownModule():
    text.set_frontend(os.environ.get("TINY_VOCA_FRONTEND", "espeak"))

try:
    import dizione
    from dizione.engine import default_directory
    HAVE_DIZIONE = (default_directory() / "lexicon.tsv").exists()
except ImportError:
    dizione, HAVE_DIZIONE = None, False

try:
    import g2pk  # noqa: F401
    HAVE_G2PK = True
except ImportError:
    HAVE_G2PK = False

READY = HAVE_DIZIONE and HAVE_G2PK
FINAL_IEUNG = text.SYMBOL_TO_ID["\u11bc"]          # final ㅇ, kept
KOREAN_IDS = {text.SYMBOL_TO_ID[jamo] for jamo in KOREAN_PHONES}


@unittest.skipUnless(READY, "needs g2pk and dizione's built data (DIZIONE_DATA)")
class UnifiedVocabTests(unittest.TestCase):
    def test_english_block_is_untouched_and_korean_is_appended(self):
        self.assertEqual(UNIFIED_VOCAB[:len(EN_SYMBOLS)], EN_SYMBOLS)
        self.assertEqual(len(EN_SYMBOLS), 76)
        self.assertEqual(len(UNIFIED_VOCAB), 142)
        self.assertEqual(len(CHOSEONG), 18)
        self.assertEqual(len(JUNGSEONG), 21)
        self.assertEqual(len(JONGSEONG), 27)
        self.assertEqual(len(KOREAN_PHONES), 66)
        self.assertNotIn(SILENT_INITIAL, CHOSEONG)
        self.assertIn("\u11ba", JONGSEONG)                       # final ㅇ survives
        self.assertEqual(text.SYMBOLS, UNIFIED_VOCAB)

    def test_every_english_id_is_unchanged(self):
        for index, symbol in enumerate(EN_SYMBOLS):
            self.assertEqual(text.SYMBOL_TO_ID[symbol], index)

    def test_no_korean_jamo_is_already_in_the_english_inventory(self):
        self.assertEqual(set(KOREAN_PHONES) & set(EN_SYMBOLS), set())


@unittest.skipUnless(READY, "needs g2pk and dizione's built data (DIZIONE_DATA)")
class PureKoreanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = text.phonemize("안녕하세요. 맑은 날씨가 좋아요!")

    def test_boundaries_and_lengths(self):
        self.assertEqual(self.result.symbols[0], "<bos>")
        self.assertEqual(self.result.symbols[-1], "<eos>")
        self.assertEqual(len(self.result.ids), len(self.result.stress))
        self.assertEqual(len(self.result.ids), len(self.result.symbols))

    def test_silent_initial_is_stripped_and_final_ieung_kept(self):
        self.assertNotIn(SILENT_INITIAL, self.result.symbols)
        self.assertIn(FINAL_IEUNG, self.result.ids)              # "안녕" -> ... ㅇ (jongseong)
        self.assertNotIn(SILENT_INITIAL, "".join(self.result.symbols))

    def test_korean_tokens_carry_no_stress(self):
        for symbol, level in zip(self.result.symbols, self.result.stress):
            if symbol in KOREAN_PHONES:
                self.assertEqual(level, 0, f"{symbol} carries stress {level}")

    def test_ids_are_in_range(self):
        self.assertGreaterEqual(min(self.result.ids), 0)
        self.assertLess(max(self.result.ids), len(UNIFIED_VOCAB))

    def test_nothing_is_unknown(self):
        self.assertEqual(self.result.unknown, ())

    def test_g2pk_rules_are_applied(self):
        """맑은 -> 말근 (nasalisation), 좋아요 -> 조아요 (liaison)."""
        pronounced = unicodedata.normalize("NFD", "말근 조아요").replace(SILENT_INITIAL, "")
        joined = "".join(self.result.symbols).replace("<sp>", "").replace(".", "").replace("!", "")
        for jamo in pronounced.replace(" ", ""):
            self.assertIn(jamo, joined)

    def test_units_group_one_syllable_each(self):
        for label, start, end in self.result.units:
            self.assertIsNotNone(label)
            self.assertLess(start, end)
            self.assertLessEqual(end - start, 3)
            self.assertTrue(all(symbol in KOREAN_PHONES for symbol in self.result.symbols[start:end]))


@unittest.skipUnless(READY, "needs g2pk and dizione's built data (DIZIONE_DATA)")
class PureEnglishTests(unittest.TestCase):
    def test_matches_dizione_exactly(self):
        for sentence in ("Hello, world!", "I read the book you read yesterday.",
                         "Dr. Patel paid $12.99 on 3/14."):
            with self.subTest(sentence=sentence):
                result = text.phonemize(sentence)
                expected = dizione.phonemize(sentence)
                self.assertEqual(result.symbols, expected.symbols)
                self.assertEqual(result.ids, expected.ids)
                self.assertEqual(result.stress, expected.stress)

    def test_stress_values_are_used(self):
        result = text.phonemize("Hello, world!")
        self.assertIn(1, result.stress)
        self.assertTrue(all(level in (0, 1, 2) for level in result.stress))

    def test_english_ids_are_the_low_block(self):
        result = text.phonemize("Hello, world!")
        self.assertLess(max(result.ids), len(EN_SYMBOLS))


@unittest.skipUnless(READY, "needs g2pk and dizione's built data (DIZIONE_DATA)")
class CodeMixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = text.phonemize("오늘 Python으로 API 테스트를 했어요.")

    def test_lengths_and_range(self):
        self.assertEqual(len(self.result.ids), len(self.result.stress))
        self.assertGreaterEqual(min(self.result.ids), 0)
        self.assertLess(max(self.result.ids), len(UNIFIED_VOCAB))

    def test_english_and_korean_tokens_both_present(self):
        english = [s for s in self.result.symbols if s in EN_SYMBOLS and s not in text.PUNCTUATION and s not in ("<bos>", "<eos>", "<sp>")]
        korean = [s for s in self.result.symbols if s in KOREAN_PHONES]
        self.assertTrue(english and korean)
        self.assertTrue(any(level > 0 for symbol, level in zip(self.result.symbols, self.result.stress)
                            if symbol not in KOREAN_PHONES), "English phonemes keep lexical stress")
        self.assertTrue(all(level == 0 for symbol, level in zip(self.result.symbols, self.result.stress)
                            if symbol in KOREAN_PHONES))

    def test_particle_is_glued_to_the_english_word(self):
        """"Python으로" has no word-boundary token between the English word and 을."""
        symbols = list(self.result.symbols)
        self.assertNotIn(SILENT_INITIAL, symbols)
        english = dizione.phonemize("Python")
        tail = list(english.symbols[1:-1])                        # p aɪ θ ɑː n
        start = symbols.index(tail[0])
        self.assertEqual(symbols[start:start + len(tail)], tail, "dizione spells Python, not the letters")
        following = symbols[start + len(tail)]
        self.assertIn(following, KOREAN_PHONES, "the particle's Jamo follow the English phonemes with no <sp>")

    def test_english_span_is_one_unit_without_a_label(self):
        labels = [label for label, _, _ in self.result.units]
        self.assertIn(None, labels, "an English phrase is one unlabelled unit for the aligner")

    def test_english_phrase_keeps_dizione_ids(self):
        result = text.phonemize("Python")
        expected = dizione.phonemize("Python")
        self.assertEqual(result.symbols, expected.symbols)
        self.assertEqual(result.ids, expected.ids)

    def test_english_run_inside_korean_has_no_spurious_space(self):
        result = text.phonemize("AI를")
        symbols = "".join(result.symbols)
        self.assertNotIn("<sp>", symbols)
        self.assertEqual(result.symbols[0], "<bos>")
        self.assertEqual(result.symbols[-1], "<eos>")

    def test_units_index_their_tokens_exactly(self):
        """The aligner reads ``[first, one past last)`` token indices, so a unit must
        hold its own syllable's Jamo, units must not overlap, and every phone of the
        utterance must belong to one unit."""
        for sentence in ("안녕하세요. 맑은 날씨가 좋아요!", "오늘 Python으로 API 테스트를 했어요.",
                         "Python을 실행하세요.", "Hello, world!", "와, 정말 대단하네요!"):
            with self.subTest(sentence=sentence):
                result = text.phonemize(sentence)
                symbols = result.symbols
                covered = []
                for label, start, end in result.units:
                    self.assertLess(start, end, "an unit is never empty")
                    self.assertLessEqual(end, len(symbols), "an unit runs past the last token")
                    covered.extend(range(start, end))
                    if label is not None:
                        expected = unicodedata.normalize("NFD", label).replace(SILENT_INITIAL, "")
                        self.assertEqual("".join(symbols[start:end]), expected,
                                         f"{label} does not own its own Jamo")
                self.assertEqual(covered, sorted(set(covered)), "units overlap")
                phones = {index for index, symbol in enumerate(symbols)
                          if symbol in text.PHONE_SET or symbol == "<unk>"}
                self.assertTrue(phones <= set(covered), f"phones outside any unit: {phones - set(covered)}")


@unittest.skipUnless(READY, "needs g2pk and dizione's built data (DIZIONE_DATA)")
class NormalizationTests(unittest.TestCase):
    def test_digits_are_read_in_korean(self):
        from tiny_voca.korean import expand_numbers
        self.assertEqual(expand_numbers("2026년"), "이천이십육년")
        self.assertEqual(expand_numbers("10시 30분"), "열시 삼십분")
        self.assertEqual(expand_numbers("3시"), "세시")            # hours are counted natively
        self.assertEqual(expand_numbers("오후 5시 5분"), "오후 다섯시 오분")
        self.assertEqual(expand_numbers("12시"), "열두시")
        self.assertEqual(expand_numbers("3.5"), "삼점오")
        self.assertEqual(expand_numbers("100원"), "백원")

    def test_quotes_and_whitespace_are_normalized(self):
        from tiny_voca.korean import normalize_ko
        self.assertEqual(normalize_ko("  안녕하세요   —   잘   지내?  "), "안녕하세요 — 잘 지내?")

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            text.phonemize("   ")

    def test_punctuation_survives(self):
        result = text.phonemize("가세요. 오세요! 네?")
        self.assertIn(".", result.symbols)
        self.assertIn("!", result.symbols)
        self.assertIn("?", result.symbols)


if __name__ == "__main__":
    unittest.main()
