"""The bilingual ``--serve-frontend`` flow of ``tiny_voca.evaluate`` (v2 English gate).

A bilingual checkpoint is trained under the ko-loan front end (142 tokens) but must
speak the English corpus under the own front end (76 tokens). The gate loads the
checkpoint under the training front end and switches the serving front end after
data loading. These tests pin the order, the failure mode that motivated it, and
the ASR defaults the gate relies on.
"""

import tempfile
import unittest

import torch

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.infer import load_model, speak
from tiny_voca.model import TinyVoca
from tiny_voca.train import export_model

try:
    import dizione
    from dizione.engine import default_directory
    HAVE_DIZIONE = (default_directory() / "lexicon.tsv").exists()
except ImportError:
    dizione, HAVE_DIZIONE = None, False

BILINGUAL_SPEAKERS = ("KSS", "F2", "M2")
EN_TEXT = "It was a bright cold day in April, and the clocks were striking thirteen."
KO_TEXT = "안녕하세요. 오늘 날씨가 정말 맑고 좋네요."


def bilingual_model():
    """A small bilingual checkpoint's model: 3 speakers, the 142-token inventory,
    duration and vocoder weights pinned so synthesis is finite (test_export pattern)."""
    torch.manual_seed(0)
    model = TinyVoca(Config(speakers=BILINGUAL_SPEAKERS, vocab_size=142)).eval()
    torch.nn.init.zeros_(model.duration.out.weight)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


class ServeFrontEndTests(unittest.TestCase):
    def tearDown(self):
        text.set_frontend("espeak")

    def test_config_vocab_validates_against_active_frontend(self):
        """The failure mode behind --serve-frontend: a 142-token config can only be
        built while a 142-token front end (ko/ko-loan) is active."""
        text.set_frontend("ko-loan")
        self.assertEqual(Config(speakers=BILINGUAL_SPEAKERS, vocab_size=142).vocab_size, 142)
        text.set_frontend("own")
        with self.assertRaises(ValueError):
            Config(speakers=BILINGUAL_SPEAKERS, vocab_size=142)

    @unittest.skipUnless(HAVE_DIZIONE, "needs dizione and its built data (DIZIONE_DATA)")
    def test_bilingual_checkpoint_loads_under_training_frontend_only(self):
        with tempfile.TemporaryDirectory() as directory:
            text.set_frontend("ko-loan")
            path = f"{directory}/bilingual.pt"
            export_model(bilingual_model(), path, step=0)
            text.set_frontend("own")
            with self.assertRaises(ValueError):
                load_model(path, check_frontend=False)   # Config cannot even be built under own
            text.set_frontend("ko-loan")
            model = load_model(path)
            self.assertEqual(model.config.speakers, BILINGUAL_SPEAKERS)
            self.assertEqual(model.config.vocab_size, 142)

    @unittest.skipUnless(HAVE_DIZIONE, "needs dizione and its built data (DIZIONE_DATA)")
    def test_serve_frontend_switch_after_load(self):
        """The EN gate order: load under the training front end, switch to own,
        synthesise. own's 76 tokens are the id-prefix of the 142 inventory, so the
        English ids are valid in the bilingual model, and the switch reverses."""
        with tempfile.TemporaryDirectory() as directory:
            text.set_frontend("ko-loan")
            path = f"{directory}/bilingual.pt"
            export_model(bilingual_model(), path, step=0)
            model = load_model(path)
            text.set_frontend("own")
            wave, phones, _ = speak(model, EN_TEXT, "F2")
            self.assertTrue(len(wave) > 0)
            self.assertTrue(torch.isfinite(wave).all())
            self.assertTrue(all(0 <= i < 76 for i in phones.ids))
            self.assertEqual(model.config.speakers.index("F2"), 1)
            # reversing the switch gives the Korean serving mode again
            text.set_frontend("ko-loan")
            wave_ko, phones_ko, _ = speak(model, KO_TEXT, "KSS")
            self.assertTrue(len(wave_ko) > 0)
            self.assertTrue(any(i > 75 for i in phones_ko.ids))

    @unittest.skipUnless(HAVE_DIZIONE, "needs dizione and its built data (DIZIONE_DATA)")
    def test_modes_phonemise_differently(self):
        """The browser worker routes each mode through set_frontend + phonemize;
        the two modes must produce different id ranges on their own corpora."""
        text.set_frontend("ko-loan")
        ko = text.phonemize(KO_TEXT)
        self.assertTrue(any(i > 75 for i in ko.ids))      # Korean NFD Jamo beyond the English inventory
        text.set_frontend("own")
        en = text.phonemize(EN_TEXT)
        self.assertTrue(all(0 <= i < 76 for i in en.ids))  # the English inventory only

    def test_asr_defaults_follow_the_serving_frontend(self):
        from tiny_voca.evaluate import asr_defaults
        self.assertEqual(asr_defaults("ko-loan"), ("small", "ko"))
        self.assertEqual(asr_defaults("ko"), ("small", "ko"))
        self.assertEqual(asr_defaults("own"), ("small.en", "en"))
        self.assertEqual(asr_defaults("espeak"), ("small.en", "en"))


if __name__ == "__main__":
    unittest.main()
