"""Phase 1 gate for the v2.0 bilingual front end (V2_SPEC.md §2.4).

The golden set below is the contract between the front end and the rest of
the pipeline.  Every reading was verified against the locked tables
(V2_SPEC.md §2.2), g2pK's pronunciation model, and the loanword dictionary.
If a case fails, either the implementation drifted or the case is wrong:
fix whichever side is linguistically incorrect and record the decision in
WORKLOG.md.  A gate that is ever "fixed" by silently changing a reading
without a log entry defeats the point.

Case groups
-----------
* ``TNTests`` (75): ``korean_normalize`` — pure text normalization; needs
  no acoustic engine.
* ``LoanwordTests`` (52): English words read as Korean loanwords in Mode A
  (``ko-loan``); the joined unit labels must equal the dictionary
  loanspelling exactly.
* ``SandhiTests`` (21): Mode A Korean spans must equal g2pK's pronunciation
  output (liaison, nasalisation, tensification, ㅎ-deletion, ...).  Note that
  a few readings are g2pK model quirks (e.g. 십이점 -> 시비점); the v1
  annotations were produced by the same model, so the gate pins them.
* ``ModeASegTests`` (22): whole utterances through ``phonemize_mixed`` —
  the exact unit-label tuple, zero ``<unk>``, zero surviving digits.
* ``ModeBTests`` (12): the ``own`` front end stays byte-identical to
  dizione.

182 golden cases in total (>= the 120 required by §2.4).
"""

import os
import unittest
from pathlib import Path

os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")
if "DIZIONE_DATA" not in os.environ:
    for candidate in (Path(__file__).resolve().parents[2] / "weights" / "phonemizer",
                      Path(__file__).resolve().parents[1] / "weights" / "phonemizer"):
        if (candidate / "lexicon.tsv").exists():
            os.environ["DIZIONE_DATA"] = str(candidate)
            break

from tiny_voca import text  # noqa: E402
from tiny_voca.korean import korean_normalize, phonemize_mixed  # noqa: E402

try:
    import g2pk  # noqa: F401
    HAVE_G2PK = True
except ImportError:
    HAVE_G2PK = False

try:
    import dizione
    from dizione.engine import default_directory
    HAVE_DIZIONE = (default_directory() / "lexicon.tsv").exists()
except ImportError:
    dizione, HAVE_DIZIONE = None, False


# ---------------------------------------------------------------------------
# TN: (input, locked reading) — V2_SPEC.md §2.2, probe-verified 75/75
# ---------------------------------------------------------------------------

TN_CASES = [
    # native counters (1-20 native, 20+ Sino): number and counter unspaced
    ("3개", "세개"), ("3명", "세명"), ("3살", "세살"), ("3마리", "세마리"),
    ("3장", "세장"), ("3통", "세통"), ("3팩", "세팩"), ("3권", "세권"),
    ("3잔", "세잔"), ("3끼", "세끼"), ("3칸", "세칸"), ("3번", "세번"),
    ("3방", "세방"), ("3수", "세수"), ("3술", "세술"), ("3짝", "세짝"),
    ("3채", "세채"), ("3척", "세척"),
    ("3세", "삼 세"), ("3박", "삼 박"), ("20살", "스무 살"),
    # Sino counters: Sino numeral, unspaced
    ("3층", "삼층"), ("3년", "삼년"), ("3일", "삼일"), ("3분", "삼분"),
    ("3초", "삼초"), ("3원", "삼원"), ("3회", "삼회"), ("3등", "삼등"),
    ("3점", "삼점"), ("3호", "삼호"), ("3기", "삼기"), ("3차", "삼차"),
    ("3페이지", "삼페이지"), ("3코어", "삼코어"),
    # hours: 1-12 native, 13-24 Sino, 0 = 영 시 (spaced)
    ("3시 30분", "세시 삼십분"), ("14:30", "십사시 삼십분"),
    ("13:15", "십삼시 십오분"), ("20:00", "이십시"), ("0시", "영 시"),
    ("12시", "열두시"), ("10시 30분", "열시 삼십분"), ("오후 5시 5분", "오후 다섯시 오분"),
    # dates: month names 유월/시월, day = Sino numeral + 일
    ("2026년 3월 14일", "이천이십육년 삼월 십사일"),
    ("6월", "유월"), ("10월", "시월"),
    ("3/14", "삼월 십사일"), ("14/3", "삼 분의 십사"),
    ("2026/3/14", "이천이십육년 삼월 십사일"), ("2026-03-14", "이천이십육년 삼월 십사일"),
    # phone numbers: digit-by-digit (0 -> 공), groups of 3-4-4
    ("010-1234-5678", "공일공 일이삼사 오육칠팔"),
    ("01012345678", "공일공 일이삼사 오육칠팔"),
    # codes: digit-by-digit when 3+ digits
    ("123번", "일이삼번"), ("402호", "사공이호"),
    # ordinal: native numeral + 번째
    ("3번째", "세 번째"),
    # percent: Sino numeral + 퍼센트
    ("5%", "오 퍼센트"),
    # currency: Sino numeral + unit word (+ 센트)
    ("$12.99", "십이 달러 구십구 센트"), ("$12", "십이 달러"),
    ("₩3000", "삼천 원"), ("€20", "이십 유로"), ("£5", "오 파운드"), ("¥100", "백 엔"),
    # range: A에서 B (+ counter)
    ("100~200명", "백에서 이백 명"),
    # fixed expression
    ("1+1", "원플러스원"),
    # decimals: A점B, digit-by-digit (0 -> 영)
    ("3.5", "삼점오"), ("3.05", "삼점영오"),
    # bare digit runs: Sino numeral (0 -> 영)
    ("123", "백이십삼"),
    # measurement units: Sino numeral + Koreanized unit name
    ("10km", "십 킬로미터"), ("2kg", "이 킬로그램"),
    ("500ml", "오백 밀리리터"), ("60°C", "육십 도"),
    # century: Sino numeral, unspaced
    ("19세기", "십구세기"), ("20세기", "이십세기"),
    # v1 expand_numbers regressions
    ("2026년", "이천이십육년"), ("100원", "백원"),
]


# ---------------------------------------------------------------------------
# Loanwords: (english word, dictionary loanspelling) — probe-verified
# ---------------------------------------------------------------------------

LOANWORD_CASES = [
    ("hello", "헬로"), ("world", "월드"), ("quick", "퀵"), ("internet", "인터넷"),
    ("computer", "컴퓨터"), ("things", "싱스"), ("yellow", "옐로우"), ("fire", "파이어"),
    ("water", "워터"), ("color", "컬러"), ("power", "파워"), ("price", "프라이스"),
    ("office", "오피스"), ("fish", "피시"), ("month", "먼스"), ("shop", "십"),
    ("shopper", "쇼퍼"), ("smart", "스마트"), ("dollar", "달러"), ("python", "파이썬"),
    ("korean", "코리안"), ("coffee", "커피"), ("english", "잉글리시"), ("thirty", "서티"),
    ("city", "시티"), ("again", "어게인"), ("about", "어바웃"), ("xylophone", "자일로폰"),
    ("algorithm", "알고리즘"), ("quantum", "쿼텀"), ("jeep", "지프"), ("jeans", "진스"),
    ("name", "네임"), ("seven", "세븐"), ("vision", "비전"), ("chocolate", "초콜릿"),
    ("church", "처치"), ("jam", "잼"), ("bus", "버스"), ("set", "셋"),
    ("bad", "배드"), ("dog", "독"), ("cat", "캣"), ("bag", "백"),
    ("hat", "햇"), ("red", "레드"), ("head", "헤드"), ("gymnasium", "짐나지엄"),
    ("digital", "디지털"), ("bottle", "보틀"), ("ballad", "발라드"),
    ("nameplate", "네임플레이트"), ("clock", "클록"),
    ("USB", "유에스비"), ("AI", "에이아이"),
]


# ---------------------------------------------------------------------------
# Sandhi: (korean input, g2pK pronunciation) — probe-verified against G2p()
# ---------------------------------------------------------------------------

SANDHI_CASES = [
    ("맑은 날씨가 좋아요", "말근 날씨가 조아요"),
    ("밥을 먹겠습니다", "바블 먹껟씀니다"),
    ("할 일을 합니다", "하 리를 함니다"),
    ("가까운 곳으로 갑니다", "가까운 고스로 감니다"),
    ("학교에 가요", "학꾜에 가요"),
    ("좋아요", "조아요"),
    ("읽다", "익따"),
    ("안녕하세요", "안녕하세요"),
    ("세상사가", "세상사가"),
    ("국수를 먹었습니다", "국쑤를 머걷씀니다"),
    ("집에 갔어요", "지베 가써요"),
    ("물건을 샀습니다", "물거늘 삳씀니다"),
    ("다녀왔습니다", "다녀왇씀니다"),
    ("먹고 싶다", "먹꼬 십따"),
    ("보고 싶다", "보고 십따"),
    ("가까이 가다", "가까이 가다"),
    ("국회의원", "구쾨의원"),
    ("바쁘다", "바쁘다"),
    ("앉다", "안따"),
    ("알다", "알다"),
    ("좋다", "조타"),
]


# ---------------------------------------------------------------------------
# Mode A segmentation: (utterance, exact unit-label tuple) — probe-verified
# ---------------------------------------------------------------------------

MODE_A_CASES = [
    ("Python을", ("파", "이", "썬", "을")),
    ("Hello!", ("헬", "로")),
    ("AI를", ("에", "이", "아", "이", "를")),
    ("10km를", ("십", "킬", "로", "미", "터", "를")),
    ("0시는요", ("영", "시", "느", "뇨")),
    ("3번째에요", ("세", "번", "째", "에", "요")),
    ("5%입니다", ("오", "퍼", "센", "트", "임", "니", "다")),
    ("2026년에요", ("이", "처", "니", "시", "븅", "녀", "네", "요")),
    ("USB를 쓰고요", ("유", "에", "스", "비", "를", "쓰", "고", "요")),
    ("010-1234-5678이에요", ("공", "일", "공", "이", "리", "삼", "사", "오",
                            "육", "칠", "파", "리", "에", "요")),
    ("100~200명이요", ("배", "게", "서", "이", "뱅", "명", "이", "요")),
    ("안녕하세요, Hello!", ("안", "녕", "하", "세", "요", "헬", "로")),
    ("오늘은 3월 14일입니다", ("오", "느", "른", "사", "뭘", "십", "싸", "이",
                              "림", "니", "다")),
    ("19세기에요", ("십", "꾸", "세", "기", "에", "요")),
    ("20살이에요", ("스", "무", "사", "리", "에", "요")),
    ("3세입니다", ("삼", "세", "임", "니", "다")),
    ("12.99예요", ("시", "비", "점", "구", "구", "예", "요")),
    ("6월에요", ("유", "워", "레", "요")),
    ("10월에요", ("시", "워", "레", "요")),
    ("1+1이에요", ("원", "플", "러", "스", "워", "니", "에", "요")),
    ("3.5예요", ("삼", "저", "모", "예", "요")),
    ("123번이에요", ("이", "리", "삼", "버", "니", "에", "요")),
]


# ---------------------------------------------------------------------------
# Mode B (own): sentences that must phonemize byte-identical to dizione
# ---------------------------------------------------------------------------

MODE_B_CASES = [
    "Hello, it costs $3.50.",
    "The quick brown fox jumps over the lazy dog",
    "One two three four five six seven",
    "Python is a programming language",
    "It is now twelve o'clock in the morning",
    "This is a test of the unified front end",
    "Good morning, how are you today?",
    "Please speak a little more slowly",
    "The weather is nice and warm outside",
    "She sells seashells by the seashore",
    "I have forty-two apples in the basket",
    "Can you repeat that sentence for me?",
]


def _attach(cls, cases, check, condition=None, reason=""):
    """One test method per data row; ``condition`` skips them all when false."""
    skip = unittest.skipUnless(condition, reason) \
        if condition is not None else (lambda function: function)
    for index, case in enumerate(cases):
        def test_method(self, case=case):
            check(self, case)
        test_method.__name__ = f"test_{index:03d}"
        setattr(cls, test_method.__name__, skip(test_method))


class TNTests(unittest.TestCase):
    """korean_normalize matches the locked reading table exactly."""

    def _check(self, case):
        source, want = case
        got = korean_normalize(source)
        self.assertEqual(got, want, f"korean_normalize({source!r})")
        self.assertFalse(any(ch.isdigit() for ch in got), f"digit survived: {got!r}")


class LoanwordTests(unittest.TestCase):
    """Mode A English spans: joined unit labels == dictionary loanspelling."""

    def _check(self, case):
        word, spelling = case
        result = phonemize_mixed(word, "ko-loan")
        labels = [label for label, _, _ in result.units if label]
        self.assertEqual("".join(labels), spelling, f"phonemize_mixed({word!r})")
        self.assertEqual(result.unknown, (), word)
        self.assertTrue(all(0xAC00 <= ord(label) <= 0xD7A3 for label in labels), word)


class SandhiTests(unittest.TestCase):
    """Mode A Korean spans equal g2pK's pronunciation output."""

    def _check(self, case):
        source, g2pk_reading = case
        result = phonemize_mixed(source, "ko-loan")
        labels = "".join(label for label, _, _ in result.units if label)
        self.assertEqual(labels, g2pk_reading.replace(" ", ""),
                         f"phonemize_mixed({source!r})")


class ModeASegTests(unittest.TestCase):
    """Whole Mode A utterances: exact labels, no <unk>, no digits."""

    def _check(self, case):
        source, want_labels = case
        result = phonemize_mixed(source, "ko-loan")
        labels = tuple(label for label, _, _ in result.units if label)
        self.assertEqual(labels, want_labels, f"phonemize_mixed({source!r})")
        self.assertNotIn("<unk>", result.symbols, source)
        self.assertFalse(any(symbol.isdigit() for symbol in result.symbols), source)


class ModeBTests(unittest.TestCase):
    """The ``own`` front end stays byte-identical to dizione."""

    def setUp(self):
        self._previous = text.frontend()
        text.set_frontend("own")

    def tearDown(self):
        text.set_frontend(self._previous)

    def _check(self, sentence):
        result = text.phonemize(sentence)
        expected = dizione.phonemize(sentence)
        self.assertEqual((result.symbols, result.stress),
                         (expected.symbols, expected.stress), sentence)
        self.assertEqual(result.unknown, (), sentence)


_attach(TNTests, TN_CASES, lambda self, case: self._check(case))
_attach(LoanwordTests, LOANWORD_CASES, lambda self, case: self._check(case),
        condition=HAVE_G2PK, reason="needs g2pK")
_attach(SandhiTests, SANDHI_CASES, lambda self, case: self._check(case),
        condition=HAVE_G2PK, reason="needs g2pK")
_attach(ModeASegTests, MODE_A_CASES, lambda self, case: self._check(case),
        condition=HAVE_G2PK, reason="needs g2pK")
_attach(ModeBTests, MODE_B_CASES, lambda self, case: self._check(case),
        condition=HAVE_DIZIONE,
        reason="needs dizione and its built data (DIZIONE_DATA)")


if __name__ == "__main__":
    unittest.main()
