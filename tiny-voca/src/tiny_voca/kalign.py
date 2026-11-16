"""Korean forced alignment: CTC at syllable granularity, Jamo suballocation.

The recogniser is ``kresnik/wav2vec2-large-xlsr-korean``: wav2vec 2.0 XLSR fine-
tuned with CTC on Common Voice Korean.  Its 1,205 labels are Korean syllable
blocks, a word delimiter ``|`` and ``[UNK]``/``[PAD]``; the CTC blank is 1204.
Syllables are the natural unit for it - the Jamo inventory of the front end is
finer than anything the recogniser was trained on, and a Jamo-level Korean CTC
model is much weaker at predicting them than at predicting whole syllables.

So alignment happens in two steps:

1. **CTC Viterbi over the syllable sequence** of the utterance, on the
   recogniser's 20 ms grid.  This is the acoustic evidence: it is what a real
   recogniser heard in the recording, not a uniform split.
2. **Suballocation inside each syllable.**  A Korean syllable is
   ``[choseong] + jungseong + [jongseong]``, and its energy contour has exactly
   that shape: a rise into the vowel and a fall out of it.  The loudest frame of
   the syllable is the nucleus; the nucleus region is every frame that stays
   within ``nucleus_drop_db`` of it; what lies before it is the onset consonant
   and what lies after it is the coda.  Each Jamo therefore gets the frames the
   recording shows for it, and every Jamo keeps at least one frame.

English phrases (the code-mixed case) have no Korean labels, so they take the
time between their Korean neighbours and share it out per phoneme, vowels
weighing more than consonants.  KSS is pure Korean, so this path only serves
free-form prompts.

Silence is handled as in :mod:`tiny_voca.align`: it is measured at acoustic
resolution and given to the pause-capable token that sits at that position, so
``sum(durations)`` is exactly the number of acoustic frames.
"""

from dataclasses import dataclass
import math
import numpy as np

from .align import (ALIGNER_CENTER, ALIGNER_RATE, ALIGNER_STRIDE, frame_energy_db, pause_carrier,
                    quality, silence_mask, token_spans)
from .text import PHONE_SET, SP, UNK, VOWEL_SET
from .vocab import CHOSEONG, JONGSEONG

ALIGNER_MODEL = "kresnik/wav2vec2-large-xlsr-korean"
ALIGNMENT_VERSION = 1
WORD = "|"                                    # the recogniser's word-delimiter label
ALIGNER_LABEL_FALLBACK = {UNK: "[UNK]"}


@dataclass(frozen=True)
class AlignmentSettings:
    onset_offset_seconds: float = -0.019   # syllable onset relative to its first emission frame
    min_pause_frames: int = 7              # ~75 ms of silence: less stays with the neighbouring Jamo
    silence_below_peak_db: float = 45.0
    silence_above_floor_db: float = 8.0
    silence_ceiling_below_peak_db: float = 30.0
    energy_window: int = 512
    nucleus_drop_db: float = 6.0           # frames this close to the syllable peak belong to the vowel
    min_nucleus_frames: int = 3

    def to_dict(self):
        return {"alignment_version": ALIGNMENT_VERSION, "aligner": ALIGNER_MODEL, **self.__dict__}


class KoreanRecognizer:
    """The pretrained Korean CTC recogniser. Loaded from the local HF cache only."""

    def __init__(self, device="cuda", name=ALIGNER_MODEL, local_files_only=True):
        import json
        from pathlib import Path
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        self.torch, self.device = torch, torch.device(device)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, local_files_only=local_files_only).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        vocabulary = Path(hf_hub_download(name, "vocab.json", local_files_only=local_files_only))
        self.vocabulary = json.loads(vocabulary.read_text(encoding="utf-8"))
        self.blank = int(self.model.config.pad_token_id)
        self.unk = int(self.vocabulary["[UNK]"])
        self.name = name
        self.revision = vocabulary.parent.name   # snapshot directory = commit hash

    def label(self, syllable):
        """CTC id of one Korean syllable block; rare syllables fall back to ``[UNK]``."""
        return int(self.vocabulary.get(syllable, self.unk))

    def log_probs(self, wave, sample_rate):
        from scipy.signal import resample_poly
        torch = self.torch
        factor = math.gcd(sample_rate, ALIGNER_RATE)
        audio = resample_poly(np.asarray(wave, dtype=np.float64), ALIGNER_RATE // factor, sample_rate // factor)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with torch.inference_mode():
            values = torch.from_numpy(audio.astype(np.float32))[None].to(self.device, self.model.dtype)
            return self.model(values).logits[0].float().log_softmax(-1).cpu().numpy()


def label_sequence(units, recognizer):
    """``(labels, owners)``: one CTC label per Korean syllable unit, and its unit index."""
    labels, owners = [], []
    for index, (label, _, _) in enumerate(units):
        if label is None:                       # an English phrase: the recogniser has no label for it
            continue
        labels.append(recognizer.label(label))
        owners.append(index)
    if not labels:
        raise ValueError("the utterance has no Korean syllables to align")
    return labels, owners


def _syllable_regions(tokens, symbols, energy, voiced, settings):
    """Split one syllable's frames into onset, nucleus and coda.

    ``energy`` is the frame level in dB over the syllable's window, ``voiced``
    the pitch tracker's decision over the same window.  Returns the frame index
    inside the window where the nucleus starts and where it ends (exclusive).
    """
    span = len(energy)
    onset = tokens[:1] if symbols[tokens[0]] in CHOSEONG else []
    coda = tokens[-1:] if len(tokens) > 1 and symbols[tokens[-1]] in JONGSEONG else []
    if len(tokens) - len(onset) - len(coda) != 1:            # e.g. an English unit
        return None
    if span < 2:
        return 0, span
    loud = energy + np.where(voiced, 3.0, 0.0)               # a voiced frame wins a tie
    peak = int(np.argmax(loud))
    threshold = energy[peak] - settings.nucleus_drop_db
    start = peak
    while start > 0 and energy[start - 1] >= threshold:
        start -= 1
    end = peak + 1
    while end < span and energy[end] >= threshold:
        end += 1
    if onset:                                                # the onset consonant needs a frame of its own
        start = max(1, start)
    if coda:
        end = min(span - 1, end)
    if end - start < settings.min_nucleus_frames:            # grow a thin vowel into the quieter edges
        grow = settings.min_nucleus_frames - (end - start)
        start = max(1 if onset else 0, start - grow // 2)
        end = min(span - 1 if coda else span, start + settings.min_nucleus_frames)
    return start, end


def _share(tokens, symbols, lo, hi, weights):
    """Even split of ``[lo, hi)`` over ``tokens``, proportional to ``weights`` (>= 1 frame each)."""
    span = hi - lo
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1.0)
    if span < len(tokens):
        raise ValueError("audio is too short for this syllable sequence")
    share = weights / weights.sum() * span
    frames = np.maximum(np.floor(share).astype(np.int64), 1)
    while frames.sum() > span:
        frames[np.argmax(frames)] -= 1
    while frames.sum() < span:
        frames[np.argmax(share - frames)] += 1
    edges, cursor = [], lo
    for count in frames:
        edges.append(cursor)
        cursor += int(count)
    return edges


def unit_windows(units, owners, spans, frames, settings, hop, sample_rate):
    """``[first, one past last)`` acoustic frames of every unit, monotone and non-empty."""
    stride = ALIGNER_STRIDE / ALIGNER_RATE
    onsets = {}
    for (first, _), unit in zip(spans, owners):
        seconds = first * stride + settings.onset_offset_seconds
        onsets.setdefault(unit, max(seconds, 0.0))
    windows, cursor = [], 0
    for index, (label, start, end) in enumerate(units):
        need = end - start
        if index in onsets:
            frame = int(round(onsets[index] * sample_rate / hop))
            cursor = max(cursor, frame)
        windows.append([cursor, None, need])
        cursor += max(need, 1)
    for index in range(len(windows) - 1):
        windows[index][1] = windows[index + 1][0]
    windows[-1][1] = frames
    for index in range(len(windows) - 1):                     # every unit keeps one frame per token
        if windows[index][1] - windows[index][0] < windows[index][2]:
            windows[index][1] = windows[index][0] + windows[index][2]
        windows[index + 1][0] = max(windows[index + 1][0], windows[index][1])
    if windows[-1][1] - windows[-1][0] < windows[-1][2]:
        raise ValueError("audio is too short for this syllable sequence")
    return windows


def assemble_ko(symbols, units, spans, owners, frames, silent, level_db, voiced, settings, hop, sample_rate):
    """Durations in acoustic frames for every token (sums to ``frames``).

    Acoustic frame ``i`` is centred on sample ``i * hop``, the convention of the
    stored mel and pitch features.
    """
    phones = [i for i, symbol in enumerate(symbols) if symbol in PHONE_SET or symbol == UNK]
    if not phones:
        raise ValueError("the utterance has no phones")
    windows = unit_windows(units, owners, spans, frames, settings, hop, sample_rate)
    keep = np.empty(len(phones), dtype=np.int64)
    position = 0
    for index, (label, start, end) in enumerate(units):
        tokens = [i for i in range(start, end) if symbols[i] in PHONE_SET or symbols[i] == UNK]
        if not tokens:
            continue
        lo, hi = windows[index][0], windows[index][1]
        if label is not None:
            regions = _syllable_regions(tokens, symbols, level_db[lo:hi], voiced[lo:hi], settings)
            if regions is None:
                edges = _share(tokens, symbols, lo, hi, [2 if symbols[i] in VOWEL_SET else 1 for i in tokens])
            else:
                nucleus_start, nucleus_end = regions
                onset = tokens[:1] if symbols[tokens[0]] in CHOSEONG else []
                coda = tokens[-1:] if len(tokens) > 1 and symbols[tokens[-1]] in JONGSEONG else []
                edges = [lo]
                if onset:
                    edges.append(lo + nucleus_start)
                edges.append(lo + nucleus_end)
                if coda:
                    edges.append(hi)
        else:
            edges = _share(tokens, symbols, lo, hi, [2 if symbols[i] in VOWEL_SET else 1 for i in tokens])
        for token, edge in zip(tokens, edges):
            keep[position] = edge
            position += 1
    keep = keep[:position]
    if position != len(phones):
        raise ValueError("alignment did not cover every phone")
    keep = np.maximum.accumulate(keep)
    for k in range(1, len(phones)):                            # one protected frame per phone
        keep[k] = max(keep[k], keep[k - 1] + 1)
    if keep[-1] > frames - 1:
        raise ValueError("audio is too short for this phone sequence")
    starts = keep.copy()
    starts[0] = 0
    token = np.empty(frames, dtype=np.int64)
    bounds = np.append(starts, frames)
    for k in range(len(phones)):
        token[bounds[k]:bounds[k + 1]] = phones[k]
    quiet = np.flatnonzero(silent[:keep[0]])
    if quiet.size:                                             # leading silence -> BOS
        token[:quiet[-1] + 1] = 0
    quiet = np.flatnonzero(silent[keep[-1] + 1:]) + keep[-1] + 1
    if quiet.size:                                             # trailing silence -> EOS
        token[quiet[0]:] = len(symbols) - 1
    for k in range(len(phones) - 1):                           # interior pauses -> pause-capable token
        left, right = phones[k], phones[k + 1]
        if right == left + 1:
            continue
        quiet = np.flatnonzero(silent[keep[k] + 1:keep[k + 1]]) + keep[k] + 1
        if quiet.size < settings.min_pause_frames:
            continue
        token[keep[k] + 1:quiet[0]] = left
        token[quiet[0]:quiet[-1] + 1] = pause_carrier(symbols, range(left + 1, right))
        token[quiet[-1] + 1:keep[k + 1]] = right
    if np.any(np.diff(token) < 0):
        raise ValueError("alignment is not monotonic")
    durations = np.bincount(token, minlength=len(symbols)).astype(np.int64)
    if durations.sum() != frames or np.any(durations[phones] < 1):
        raise ValueError("durations do not cover the acoustic frames")
    return durations


def align_utterance(recognizer, phonemized, wave, voiced, config, settings):
    """Return ``(durations, qc)`` for one recording or raise ``ValueError``."""
    frames = len(wave) // config.hop
    if len(wave) != frames * config.hop or len(voiced) != frames:
        raise ValueError("waveform length must be frames * hop")
    labels, owners = label_sequence(phonemized.units, recognizer)
    log_probs = recognizer.log_probs(wave, config.sample_rate)
    spans = token_spans(log_probs, labels, recognizer.blank)
    level = frame_energy_db(wave, config.hop, settings.energy_window)
    silent, threshold = silence_mask(level, settings)
    voiced = np.asarray(voiced) > 0.5
    durations = assemble_ko(phonemized.symbols, phonemized.units, spans, owners, frames,
                            silent, level, voiced, settings, config.hop, config.sample_rate)
    qc = quality(phonemized.symbols, durations, spans, log_probs, labels, voiced, level, config.hop, config.sample_rate)
    qc["silence_threshold_db"] = round(threshold, 1)
    qc["syllables"] = len(labels)
    return durations.tolist(), qc
