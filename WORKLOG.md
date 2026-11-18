# WORKLOG — Korean + code-mixed TTS on `tiny-voca` + `dizione`

Workspace: `C:\ai-workspace\korean-tts` (not a git repo at the root; each clone is its own repo).
Goal: a verified 24 kHz Korean (and KO/EN code-mixed) TTS pipeline built on
[`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca) +
[`n4ze3m/dizione`](https://github.com/n4ze3m/dizione), trained on
[`Bingsu/KSS_Dataset`](https://huggingface.co/datasets/Bingsu/KSS_Dataset) resampled to 24 kHz.

KSS licence: **CC BY-NC-SA 4.0** — non-commercial research/testing only.

---

## Phase 1 — Workspace bootstrap & environment audit ✅

### Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB, driver 610.47 |
| CPU | AMD Ryzen 9 5900X 12-core, 24 logical processors |
| RAM | 42 854 653 952 B ≈ 39.9 GiB |
| Disk (C:) | ~16.7 GB free at the start of Phase 4 (tight — see storage budget) |
| OS | Windows 11 (win32), shell = PowerShell |

System tools: `git` 2.51, `ffmpeg` present, `cmake` present, `uv` present, **`sox` missing**
(not needed: resampling is done with `scipy.signal.resample_poly`, I/O with `soundfile`).

Python: `3.11.16 (MSC v.1944 64 bit)` in `./.venv` (created with `uv venv`).

### Environment variables every command needs

```powershell
$env:PYTHONIOENCODING = "utf-8"          # console codepage is cp949
$env:PYTHONUTF8       = "1"
$env:DIZIONE_DATA     = "C:\ai-workspace\korean-tts\weights\phonemizer"
$env:TINY_VOCA_FRONTEND = "ko"           # must be set BEFORE importing tiny_voca
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

`TINY_VOCA_FRONTEND` matters because `text.py` binds `SYMBOLS`, `PHONE_SET`, `VOWELS` at
import time and `data.py`/`model.py` import those module-level names.

The venv has **no `pip`**; install with `uv pip install …` after
`$env:VIRTUAL_ENV = "C:\ai-workspace\korean-tts\.venv"`. The global `hf` CLI is broken;
use `.\.venv\Scripts\hf.exe`.

### `dizione` architecture (English front end)

```
src/dizione/__init__.py   phonemize(text) -> Phonemized(symbols, ids, stress, unknown)
        engine.py         dictionary lookup + G2P + homographs, data directory
        symbols.py        the 76-symbol inventory
        g2p.py            char-ngram grapheme->phoneme model for unknown words
        lexicon/homographs/morphology/normalize/sentence/convert/align
```

* Data directory: `DIZIONE_DATA`, else `./weights/phonemizer`. Built once with
  `python -m dizione.build` (see Phase 2); our build lives in `weights/phonemizer/`
  (`lexicon.tsv`, `g2p.npz`, `align.json`, `homographs.json`), fingerprint
  `8663b0cf7529c14356544f6eccac657c58b4614bc534f517b6bd5586a5c76c4d`.
* **Boundary tokens:** `phonemize()` **does** wrap the utterance in `<bos>`/`<eos>`.
  Verified: `dizione.phonemize("Hello, world!")` →
  `<bos> h ɛ l oʊ , w ɜː l d ! <eos>`, ids `(1,31,43,35,62,5,38,58,35,20,8,2)`,
  stress `(0,0,0,0,1,0,0,1,0,0,0,0)`. Word boundaries inside a phrase are `<sp>`.
* Inventory (76, indices 0..75): 5 specials `<pad> <bos> <eos> <sp> <unk>`,
  12 punctuation `, . ? ! ; : " ' ( ) — …`, 26 consonants, 33 vowels.

### `tiny-voca` architecture

```
src/tiny_voca/
  vocab.py     NEW  token inventories: EN_SYMBOLS (76) + 66 NFD Jamo = UNIFIED_VOCAB (142)
  text.py      front-end switch espeak | own | ko; Phonemized(symbols, ids, stress, unknown, units)
  korean.py    NEW  phonemize_mixed(): code-mixed KO(NFD Jamo) + EN(dizione)
  config.py    Config dataclass; vocab_size 0 -> len(SYMBOLS) of the active front end
  prepare.py   Parquet/JSONL -> wave/mel/prosody .pt files + <split>.jsonl manifest
  align.py     English CTC aligner (facebook/wav2vec2-lv-60-espeak-cv-ft)
  kalign.py    NEW  Korean CTC aligner (kresnik/wav2vec2-large-xlsr-korean)
  annotate.py  phonemize + forced align + QC gate -> annotations/<split>.jsonl
  data.py      Dataset/collate; the annotation invariants below
  model.py     the ~5 M-parameter TTS model; losses.py; train.py; refine.py
  infer.py     checkpoint + text -> .wav; evaluate.py; export.py (ONNX); journal.py
```

### Exact schemas (read from the source, not assumed)

**`prepare.py` input Parquet** (`--text-column text`, `--speaker-column voice_id`):

| column | required | notes |
| --- | --- | --- |
| `id` | yes | stable non-empty string, unique |
| `split` | yes | must equal `--split` |
| `text` | yes | non-empty string |
| `voice_id` | yes | must be in `Config.speakers` |
| `audio` | yes | `str` path, `{"path": …}` (relative to the Parquet's parent) **or** `{"bytes": …}` |
| `text_id`, `origin`, `source_id`, `source_license`, `source_url`, `teacher_model`, `reference_audio` | optional | copied into `provenance` |

Audio must **already** be 24 000 Hz mono (`sf.read` then `sr != 24000` → rejected).
Admission: `32 <= ceil(len(wave)/hop) <= 3000` frames, `max|wave| <= 1.01`, RMS > 1e-5.

**Feature file** `features/<split>/<sha256(id)>.pt`: `wave` (float32 `[n*hop]`, zero-padded to
whole frames), `mel` (float32 `[frames, 80]`, log-mel), `prosody` (float32 `[frames, 3]` =
`log(f0/200)` where voiced, voicing 0/1, `log(rms)`), `speaker` (int index into `Config.speakers`).
Acoustic grid: `sample_rate 24000`, `hop 256` (10.667 ms/frame, 93.75 frames/s), `n_fft 1024`,
`n_mels 80`, `f0 50–600 Hz`.

**`annotations/<split>.jsonl`** (written by `annotate.py`, read by `data.py`):
`id, split, speaker, text, text_hash, path, frames, symbols` (space-joined), `tokens`,
`stress`, `durations`, `unknown`, `qc`, `provenance`, `acoustic`, `frontend`, `annotation`.

Invariants enforced in `data.py`:

* `len(tokens) == len(durations) == len(stress)`
* `sum(durations) == frames` (frames = `len(wave) // hop`)
* `min(durations) >= 0`, every *phone* token has `>= 1` frame
* `len(tokens) <= max_tokens (512)`, `max(tokens) < vocab_size`, `min(tokens) >= 1`

**`annotations/phonemes.jsonl`** (front-end cache, one row per distinct text):
`text_hash, symbols, stress, unknown, units` — `units` is our addition:
`[label, first_token, one_past_last]`, where `label` is the Korean syllable block the
acoustic aligner knows, or `None` for an English phrase.

### Exact CLI (observed from the `argparse` blocks)

```
python -m tiny_voca.prepare    --inputs A.parquet [...] --output DIR --split train|validation|test
                               [--text-column text] [--speaker-column voice_id] [--config f.json] [--resume]
python -m tiny_voca.annotate   --features DIR --output DIR --dataset-revision REV
                               [--config f.json] [--device cuda] [--splits ...] [--limit N] [--resume]
                               [--refine N] [--refine-band 4] [--stage ctc|refined-N]
                               [--min-pause-frames 7] [--max-weak-phone-fraction 0.25]
                               [--min-token-probability 0.5] [--max-phone-frames 70]
                               [--min-vowel-voiced 0.5] [--min-phones-per-second 4.0]
                               [--max-phones-per-second 30.0]
python -m tiny_voca.train      --train train.jsonl --validation val.jsonl --output DIR
                               [--config f.json] [--device cuda] [--steps 100000] [--batch-frames 9000]
                               [--max-batch 32] [--learning-rate 3e-4] [--warmup 1000]
                               [--lr-half-life 100000] [--grad-clip 10.0] [--weights JSON]
                               [--gan-start 8000] [--gan-ratio 0.7] [--gan-ramp 3000]
                               [--feature-matching 30.0] [--gan-scale-max 2000.0] [--crop-samples 8192]
                               [--crops 2] [--teacher-hold 3000] [--teacher-ramp 7000] [--teacher-final 0.0]
                               [--subset N] [--validation-size 32] [--save-every 1000] [--keep-every 10000]
                               [--log-every 10] [--seed 20261113] [--max-seconds 3600]
                               [--deadline-file f] [--deadline-margin 300] [--resume CKPT]
                               [--init-from CKPT] [--new-data] [--stop-file f] [--recorded-voicing]
                               [--no-preload]
python -m tiny_voca.infer      --checkpoint CKPT --text "..." --output f.wav [--voice NAME] [--speed 1]
                               [--pitch-shift 0] [--pitch-range 1] [--device cpu] [--frontend ko]
                               [--allow-frontend-mismatch]
python -m tiny_voca.evaluate   --checkpoint CKPT --annotations DIR --output DIR --tag TAG
                               [--splits ...] [--train-subset N] [--per-voice 4] [--novel N]
                               [--device cuda] [--asr-model small.en] [--no-asr] [--seed 0] [--speed 1.0]
                               [--pitch-range 1.0]
```

`train.py` rejects a model with more than 5 000 000 inference parameters; the 142-token
inventory keeps us under that (checked in Phase 5).

---

## Phase 2 — Python environment & dependencies ✅

`.venv` (uv, CPython 3.11.16). Installed: torch 2.14.1+cu126, torchaudio 2.11.0+cu126,
transformers 4.57.6, datasets, pyarrow 26.0.1, librosa 0.11.0, scipy, soundfile,
g2pk + python-mecab-ko 1.3.7 + python-mecab-ko-dic, faster-whisper 1.2.1, jiwer,
onnx/onnxruntime (for `export.py`), `dizione` and `tiny-voca` in editable mode.

Gate script `scripts/verify_phase2.py` prints `PHASE2 OK`:

* `dizione.phonemize("Hello, world!")` → `<bos> h ɛ l oʊ , w ɜː l d ! <eos>` (12 ids)
* `G2p()("맑은 날씨가 좋아요.")` → `말근 날씨가 조아요.` (no MeCab/dictionary errors)

`datasets` audio decoding needs `torchcodec` (not installed), so KSS is read with
`cast_column("audio", Audio(decode=False))` + `soundfile`.

---

## Phase 3 — Unified NFD-Korean + dizione-English front end ✅

### `UNIFIED_VOCAB` (142 tokens)

| block | indices | count | contents |
| --- | --- | --- | --- |
| English (dizione, untouched) | 0–75 | 76 | 5 specials, 12 punctuation, 26 consonants, 33 vowels |
| Choseong | 76–93 | 18 | U+1100–U+1112 **excluding** silent initial U+110B |
| Jungseong | 94–114 | 21 | U+1161–U+1175 |
| Jongseong | 115–141 | 27 | U+11A8–U+11C2 (keeps final ㅇ U+11BC) |

Append-only, so every English id produced by `dizione` is unchanged.
`Config.vocab_size` defaults to `0` = "the active front end's inventory"
(76 for `espeak`/`own`, 142 for `ko`).

### `phonemize_mixed(text)` (`tiny_voca/korean.py`)

1. Whole-utterance normalisation: NFKC, quote/dash folding, ellipsis, whitespace.
2. `EN_SPAN` segmentation: a span starts with a Latin letter and continues over
   `[A-Za-z0-9''-:/]`, internal spaces/punctuation, and number tokens
   (`$12.99`, `3/14`), so `"Hello, world!"` and `"Dr. Patel paid $12.99 on 3/14."`
   each reach `dizione.phonemize` as **one contiguous phrase**.
3. English chunk → `dizione.phonemize`, `<bos>`/`<eos>` stripped, ids and stress
   `[0,1,2]` kept verbatim.
4. Korean chunk → `expand_numbers` (Sino-Korean, e.g. `2026년` → `이천이십육년`) →
   `g2pk` (liaison, nasalisation, tensification, ㅎ-deletion) → NFD → drop silent
   initial ㅇ (U+110B) → one token per Jamo/space/punctuation, `stress = 0`.
5. `<sp>` is inserted between chunks only where the source had whitespace, so
   `Python으로` becomes `p aɪ θ ɑː n` + `ㅡ ㄹ` with **no** word-boundary token between them.
6. One `<bos>`/`<eos>` pair around the whole utterance.
7. `units` records the alignable groups: one Korean syllable block per unit (label =
   the syllable the acoustic aligner knows), one English phrase per unit (label `None`).

### Gate: `tests/test_ko_en_frontend.py` — 23 tests, all pass

```
Ran 23 tests in 0.791s
OK
```

Full suite (with the Korean front end active): `Ran 66 tests ... OK (skipped=2)`.
Three pre-existing tests were made front-end aware (`test_frontend.py` pins `espeak` in
`setUp`, `test_frontend_own.py` compares against `EN_SYMBOLS`, `test_model.py` seeds the
untrained-model pitch-spread check); `onnx`/`onnxruntime` were added for `test_export.py`.

---

## Phase 4 — Dataset download & preprocessing ✅

`Bingsu/KSS_Dataset` (`load_dataset(..., split="train")`): **12 854 rows**, columns
`audio` (Audio 44 100 Hz mono), `original_script`, `expanded_script`, `decomposed_script`,
`duration` (float32 seconds), `english_translation`.

`scripts/prepare_kss.py` → `data/{train,validation,test}-00000-of-00001.parquet`:

* resample 44 100 → **24 000 Hz** mono (`resample_poly`, gcd 100 → 240/441), stored as
  PCM-16 WAV bytes inside the shard;
* `text = expanded_script` (numbers/units already spoken Korean, punctuation preserved);
* `voice_id = "KSS"`, `id`/`text_id` = the source WAV basename (e.g. `ssb1047_0001`);
* keep `0.5 s <= duration <= 15 s`, `frames <= 3000`, `len(tokens) < 512`;
* deterministic splits from `seed = 42`: 95 % / 2.5 % / 2.5 %;
* `phonemize_mixed` is run on every kept row here, so `<unk>` counts and token lengths are
  measured before `prepare` ever runs.

Storage budget: shards ≈ 2.1 GB, feature `.pt` files ≈ 5.6 GB.

### Measured on the first pass (`scripts/verify_phase4.py`)

| split | rows | hours | seconds min/max |
| --- | --- | --- | --- |
| train | 12 211 | 12.21 | 0.77 / 9.01 |
| validation | 321 | 0.32 | 1.47 / 8.17 |
| test | 322 | 0.32 | 1.44 / 7.94 |

All 12 854 KSS clips survive the filters (0.5–15 s, `< 512` tokens): **12.85 h**,
`sample_rate = 24000` everywhere, `voice_id = "KSS"`, no empty transcripts,
**zero `<unk>`** from `phonemize_mixed` over the whole corpus.

One fix during this phase: the HF rows carry `audio = {"bytes": …, "path": None}`, so the
source filename is not available and `id` came out empty (which `prepare.py` rejects).
Ids are now the stable positional `kss-00000 … kss-12853`.

---

## Phase 5 — Model config & Korean forced aligner

`config-ko.json`: `speakers: ["KSS"]`, `vocab_size: 142`, acoustics at the defaults.

### Aligner choice

| candidate | label set | CTC head | verdict |
| --- | --- | --- | --- |
| `facebook/mms-1b-all` (`kor` adapter) | 1330 (syllables + romanisation) | **154** classes | rejected: the adapter's CTC head is far coarser than its vocab |
| `kresnik/wav2vec2-large-xlsr-korean` | **1205** = 1202 Hangul syllable blocks + `\|` + `[UNK]` + `[PAD]` | 1205, blank = 1204 | **chosen** |

Weights: `model.safetensors` (1.21 GB) in the HF cache, snapshot
`629c9a3501c10ba128bf3fa1eebb12af3be03f61`.

### `tiny_voca/kalign.py` — two-step alignment

1. **CTC Viterbi at syllable granularity** (`torchaudio.functional.forced_align`, with the
   numpy `ctc_viterbi` from `align.py` as a fallback) over the syllable sequence of the
   utterance, on the recogniser's 20 ms grid. Rare syllables outside the 1202 labels use
   `[UNK]`.
2. **Jamo suballocation inside each syllable.** A Korean syllable is
   `[choseong] + jungseong + [jongseong]` and its energy contour has exactly that shape.
   The loudest (voicing-weighted) frame of the syllable is the nucleus; the nucleus region
   is every frame within `nucleus_drop_db = 6 dB` of it; frames before it belong to the
   onset consonant, frames after it to the coda. Each Jamo keeps at least one frame.

Silence is measured at acoustic resolution and given to the pause-capable token at that
position (BOS / EOS / punctuation / `<sp>`), exactly as the English aligner does, so
`sum(durations) == frames` by construction. English phrases (code-mixed prompts) have no
Korean labels: they take the time between their Korean neighbours and share it out per
phoneme, vowels weighing more than consonants.

`annotate.py` picks the aligner from the active front end:
`ko → kalign.KoreanRecognizer`, otherwise `align.PhoneRecognizer`.

### Gate, part 1 — model instantiation ✅ (`scripts/verify_phase5_model.py`)

```
frontend inventory: 142 config vocab_size: 142
inference parameters: 4,549,579 (train.py allows up to 5,000,000)
batch tokens: torch.Size([1, 47]) mel: torch.Size([1, 98, 80])
objective: 565.5091552734375 terms: ['base_loss', 'duration', ...]
PHASE5 MODEL OK  vocab 142  speakers ('KSS',)  params 4,549,579  under 5M: True
```

A forward pass on a real Korean sentence (`안녕하세요. 오늘 날씨가 정말 맑고 좋네요.`,
47 tokens → 98 frames) produces a finite objective.

### Gate, part 2 — 20-clip alignment check ✅

```
python -m tiny_voca.prepare  --inputs data_smoke/train-00000-of-00001.parquet --output features_smoke --split train --config config-ko.json
python -m tiny_voca.annotate --features features_smoke --output annotations_smoke --dataset-revision smoke-20 --splits train \
        --config config-ko.json --device cuda --max-weak-phone-fraction 0.6 --min-token-probability 0.3
python scripts/verify_phase5_align.py annotations_smoke/train.jsonl
```

```
clips: 17  failures: 0
phone frames  p0/p50/p95/max: [1.0, 6.0, 21.0, 59.0]
phones/second mean: 11.41   token posterior mean: 0.508
  token_probability_mean   p5= 0.389 p50= 0.485 p95= 0.690
  weak_phone_fraction      p5= 0.250 p50= 0.421 p95= 0.600
  max_phone_frames         p5=17.800 p50=34.000 p95=55.000
  vowel_voiced             p5= 0.743 p50= 0.883 p95= 0.973
  pause_level_db           p5=-61.460 p50=-56.400 p95=-50.360
  leading_frames           p5= 4.000 p50= 9.000 p95=24.000
  trailing_frames          p5= 2.800 p50= 4.000 p95=27.200
```

All 20 clips aligned; **17 accepted, 0 structural failures**:
`len(tokens) == len(stress) == len(durations)`, `sum(durations) == frames`, every Jamo has
`>= 1` frame, and no clip collapsed onto one long phone. Vowels sit on voiced frames
(88 % median) and the pause tokens sit in genuinely quiet frames (−56 dB vs −9.5 dB speech).

The gate thresholds were retuned for Korean: a 1 205-way syllable softmax is less peaked
than a 59-way phone softmax, so `--max-weak-phone-fraction 0.6 --min-token-probability 0.3`
(English defaults 0.25 / 0.5) keeps 85 % of clips instead of 10 %.

Three sample alignments (`token|ms`, from `scripts/verify_phase5_align.py`):

```
kss-00003  217 frames = 2315 ms   그는 원래 사람이 많았어요
  <bos>|64  ㄱ|64  ㅡ|53  ㄴ|11  ㅐ|544  <sp>|0  ㅅ|43  ㅏ|139  ㄹ|11  ㅁ|117  ㅏ|32  ㄴ|85  ㄱ|53
  <sp>|0  ㅁ|11  ㅏ|192  ㄴ|32  ㅈ|107  <sp>|0  ㅆ|181  ㅓ|53  ㅂ|149  ㅏ|320  ㅆ|11  ㅓ|11  .|0  <eos>|32

kss-00000  331 frames = 3531 ms   그는 괜찮은 척하고 있는 것 같았어
  <bos>|96  ㄱ|85  ㅡ|96  ㄴ|11  ㄴ|149  ㅏ|85  <sp>|0  ㄱ|53  ㅘ|128  ㅊ|32  ㅏ|64  ㄴ|85  ㅅ|11  ㄴ|85  ㅡ|43
  <sp>|0  ㅅ|43  ㅜ|139  ㅁ|43  ㄴ|96  ㄱ|21  ㅛ|171  ㄴ|43  ㄱ|181  <sp>|256  ㅆ|224  ㅓ|53  ㄴ|64  ㄱ|11  ㅓ|64  ㅆ|11
  <sp>|0  ㄱ|11  ㅓ|53  ㅅ|11  <sp>|0  ㅂ|43  ㅓ|43  ㄹ|11  ㄱ|32  ㅏ|11  ㅆ|53  ㅓ|96  .|0  <eos>|725
```

Onsets are short (11–64 ms), vowels long (53–544 ms), and the word-boundary token carries
the one long pause in `kss-00000` (256 ms) — the shape the aligner is meant to produce.

---

## Phase 6 — End-to-end smoke test

100 clips sliced from the train shard (+ 10 validation, 10 test):
`prepare` accepted 100/100, 10/10, 10/10; `annotate` accepted **83 train / 6 validation /
5 test** (rejections: `weak_alignment` 13, `low_confidence` 10, `overlong_phone` 3).
Training 800 steps with the GAN phase from step 200: `runs/smoke`.

### Gate ✅ (`prepare -> annotate -> train -> infer` all clean)

* `annotate` finished with `"complete": true` — it only reports that when **all three**
  splits are annotated, so the 10-clip test split had to be prepared as well.
  Adding a split to an existing run needs a **new** `--output` directory, because the
  run identity hashes the per-split source manifests.
* 800 steps in **86 s** (8.4 steps/s at `--batch-frames 4000 --max-batch 8`), no NaN,
  checkpoints `runs/smoke/{step-00000200,step-00000400,step-00000800,last,inference}.pt`.
* Losses fell monotonically through the teacher-forcing ramp and into the GAN phase:
  mel 1.29 → 0.75, duration 0.44 → 0.16, voicing accuracy 0.77 → 0.91,
  `duration_length_ratio` 0.78 → 0.96, validation predicted mel 1.24 → **0.876**.
* `tiny_voca.infer` with `runs/smoke/inference.pt`:

```
eval\smoke\smoke_01.wav   24000 Hz  1ch   3.65s  rms 0.0814  peak 0.6359  OK
eval\smoke\smoke_02.wav   24000 Hz  1ch   2.79s  rms 0.0644  peak 0.4120  OK
```

`scripts/verify_wavs.py` checks sample rate, channel count, duration window,
finite samples, RMS floor and clipping; both files pass.

**Phase 6 exit gate: PASSED.** Early intelligibility signal from the 800-step model
(`scripts/asr_check.py`, `faster-whisper small`, `language=ko`, beam 5):

```
smoke_01  ref 안녕하세요오늘날씨가정말맑고좋네요     asr 안녕하세요안녕하세요        CER 76.5 %
smoke_02  ref 지금몇시인지혹시알려주실수 있나요      asr 그거그거그거주실수 있나요   CER 56.2 %
```

Endings and particles are already right (`…주실 수 있나요`), which is what an
800-step/83-clip model should sound like. `faster-whisper` needs `cublas64_12.dll`;
`scripts/asr_check.py` adds `torch/lib` to the DLL search path before importing it.

### Fast full-pipeline check (every stage ≤ 2 min wall clock)

Requested as a single pass over the whole process, sized so no stage waits more than
two minutes. Subset 48 train / 6 validation / 6 test clips (`data_fast/`),
`config-ko.json`, GPU for annotate/train/infer:

| # | stage | command | wall | result |
| --- | --- | --- | --- | --- |
| 0 | slice | `scripts/slice_shard.py` × 3 | 4 s | 48 / 6 / 6 rows |
| 1 | `tiny_voca.prepare` | 3 splits → `features_fast/` | 44 s | 48, 6, 6 accepted, 0 rejected |
| 2 | `tiny_voca.annotate` | CTC aligner, 3 splits → `annotations_fast/` | 40 s | `"complete": true`, 44 / 4 / 6 accepted |
| 3 | `tiny_voca.train` | 700 steps, GAN from 200 → `runs/fast/` | 90 s | no NaN, `inference.pt` written |
| 4 | synthesis | `scripts/synthesize_samples.py` → `eval/fast/` | 6 s | 7 Korean `.wav` files, 2.84–5.89 s |
| 5 | audio + ASR | `scripts/verify_wavs.py`, `scripts/asr_check.py` | 38 s | all 7 OK; mean CER 119.7 % |

`--max-phone-frames 90` removes the `overlong_phone` rejections (test split 6/6 accepted,
against 5/10 at the default 70).

Two script fixes came out of this run: `verify_wavs.py` now expands shell globs, and
`asr_check.py --manifest` resolves `<name>.wav` from `--outdir` (the manifest carries
text only).

The 700-step/44-clip model is only partly intelligible — `sample_04_numbers` is
transcribed as a string of isolated jamo (`ㄴㅑㅈㅌㅏㅈㅚ…`), i.e. the syllables are
still separated, which is the expected shape this early in training.

---

## Code review before the full run

`scripts/check_units.py` (new) asserts the invariant the aligner depends on: every
`units` entry is a non-empty, non-overlapping `[first, one past last)` range of
alignable tokens, a Korean unit owns exactly its own syllable's Jamo, and every phone
of the utterance belongs to one unit. 8 texts (pure Korean, pure English, code-mixed,
digits) pass, and the check is now also a unit test
(`test_units_index_their_tokens_exactly`, 5 sentences).

Findings and fixes:

| # | finding | fix |
| --- | --- | --- |
| 1 | `expand_numbers` read the hour in Sino-Korean: `3시` → `삼시`, while Koreans count hours natively (`세시`) | `_native_korean()` (한/두/세/네/다섯…열/스물…, 1..100) is used when the unit is `시`; `10시 30분` → `열시 삼십분`, `오후 5시 5분` → `오후 다섯시 오분`, `12시` → `열두시` |
| 2 | `text.frontend_runtime()` recorded a hardcoded `"g2pk": "g2pk2"` for the Korean front end, so an annotation run claimed a package it did not use | `_g2pk_version()` reads the installed distribution: the run now records `g2pk 0.9.4`. `frontend_identity()` uses the same helper (it used to record `"unknown"`), so the front-end fingerprint changed and the annotations were regenerated |
| 3 | `kalign.unit_windows` built a `phones` list it never used and computed token counts as `sum(1 for i in range(start, end) if True)` | `need = end - start`, dead list removed |
| 4 | `verify_wavs.py` failed on a shell glob (`eval\fast\*.wav` reached `soundfile` literally) | expands each argument through `glob` |
| 5 | `asr_check.py --manifest` required a `wav` key the manifest does not have | `--outdir` resolves `<name>.wav`; the manifest stays text-only |
| 6 | an English unit spans its whole phrase, commas included, which the strict reading of the checker flagged | by design — `assemble_ko` filters a unit down to its phone tokens; the checker now allows punctuation inside a unit |

Retest after the fixes: `tests/test_ko_en_frontend.py` **24/24**, whole suite
**67 tests OK (skipped=2)**, `scripts/check_units.py` 8/8, and the fast pipeline
re-run end to end (`annotations_fast2` → `runs/fast2` → `eval/fast2`):
annotate 30 s, train 700 steps 81 s, 7 samples 6 s, audio+ASR 38 s,
**mean CER 93.5 %** (was 119.7 % on the previous 700-step run).

---

## Phase 7 — training budget on the RTX 4090 (18 GB usable, 18 h wall clock)

`scripts/parse_train_log.py` (new) reads a train log and reports the *instantaneous*
steps/s and audio-seconds/s between records; `nvidia-smi` was sampled every 2 s while
each probe ran (44-clip fast set, 300 steps, GAN phase entered at step 100):

| `--batch-frames` / `--max-batch` | peak VRAM | steady steps/s | audio-s/s | note |
| --- | --- | --- | --- | --- |
| 4 000 / 8 (smoke) | ~9 GB | 8.4 | 358 | pre-GAN phase |
| 12 000 / 48 | **15 458 MiB** | 2.9–3.2 | **372** | chosen |
| 14 000 / 56 | 17 335 MiB | 1.95 | 291 | slower, more VRAM |
| 16 000 / 64 | 18 589 MiB | – | – | OOM inside the GAN phase |

Throughput is flat in batch size (≈360 audio-s/s at 4 000 and at 12 000 frames), so the
generator is launch/data-bound rather than compute-bound: 12 000 frames per step buys
3× the audio per step at the same speed, which is what a fixed wall-clock budget wants.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to keep the GAN phase from
fragmenting the pool.

**Chosen recipe** (`runs/full`, ~10.1 h of audio, 12 000 frames ≈ 128 s per batch):

```
python -m tiny_voca.train --train annotations_full/train.jsonl --validation annotations_full/validation.jsonl \
       --output runs/full --config config-ko.json --device cuda \
       --steps 104000 --batch-frames 12000 --max-batch 48 --learning-rate 3e-4 --warmup 1000 \
       --gan-start 8000 --gan-ramp 3000 --gan-ratio 0.7 --feature-matching 30 \
       --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 \
       --validation-size 32 --save-every 2000 --keep-every 10000 --log-every 50 \
       --max-seconds 14400
```

Validation runs on every save (52 times over the run, ≈13 s each at 32 clips ≈ 11 min),
and the run is driven in 4-hour chunks with `--resume`, so a dropped session never
loses more than the current chunk.

| stage | wall clock |
| --- | --- |
| full `prepare` (12 211 + 321 + 322 clips, CPU) | ~2.0 h |
| full `annotate` (CTC aligner, GPU) | ~15 min |
| `train` 104 000 steps | ~9.6 h + 11 min validation |
| `evaluate` + 7 samples + ASR | ~20 min |
| **total** | **≈ 12.5 h** (budget 18 h) |

### Full acoustic preparation ✅

`tiny_voca.prepare --inputs data/<split>-00000-of-00001.parquet --output features --split <split> --config config-ko.json`:

| split | accepted | rejected |
| --- | --- | --- |
| train | 12 211 | 0 |
| validation | 321 | 0 |
| test | 322 | 0 |

~100 clips/min on 24 threads (librosa `pyin` dominates), 4.5 GB of `.pt` features.
The run was split over two invocations because the first hit a 2 h shell timeout at
11 599 clips; `--resume` continues from the journal, which is exactly what it is for.

`scripts/train_full.ps1` drives the training in wall-clock chunks and was tested on the
fast set (120 → 225 → 240 steps across three chunks, resuming from `last.pt` each time).

`tiny_voca.evaluate` was made Korean-aware while waiting: the ASR language follows the
front end (`ko` → Whisper `ko` instead of the hardcoded `en`), the ASR model defaults to
`small` for Korean, it runs on the GPU when the evaluation does, the text normalizer has
a Korean branch, and the report now carries **CER** next to WER (Korean is scored at
character level). Probe on the 700-step checkpoint: 4 samples, 19.6 s,
`generated_cer 4.85`, `mel_dtw 1.032`, `duration_ratio 0.885`,
`frontend_token_mismatches 0`.

### Sentence leakage → group-wise split

The first full `annotate` stopped with `ValueError: sentence leakage between train and
validation`: KSS records some sentences more than once (12 854 rows, **12 463 distinct
sentences**), and a row-wise random split puts two takes of one sentence in
different splits. `scripts/resplit_kss.py` now assigns every recording of a transcript
to one split, same seed 42, same 95 / 2.5 / 2.5 targets:

| split | rows | hours | distinct sentences |
| --- | --- | --- | --- |
| train | 12 210 | 12.211 | 11 833 |
| validation | 322 | 0.326 | 313 |
| test | 322 | 0.321 | 317 |

`prepare --resume` then only recomputes the features of the rows that changed split —
which turned out to be most of them, and the journal is keyed by the shard's sha256, so
the whole set is recomputed.

### Parallel acoustic preparation

`prepare` is single-process and `librosa.pyin` dominates, so 12.8 k clips cost ~2 h.
`scripts/split_shards.py` cuts the train shard into 4 parts and 6 processes run at once
(4 × train + validation + test) on 24 cores; `scripts/merge_features.py` stitches the
per-part directories back into one `features2/` directory with a single
`<split>.jsonl` manifest, re-keying `record_key` to `<part>:<index>` so the journal ids
stay unique. Feature `.pt` files are named by a hash of the clip id, so they never
collide. The merged metadata keeps the acoustic fingerprint, which is what
`annotate.load_acoustic_rows` checks.

Result: 3 053 × 3 + 3 051 = **12 210** train + 322 validation + 322 test, **0 rejected**,
wall clock **~22 min** for the six parallel processes (vs ~2 h single-process) plus 88 s
to merge. `features2/` is the merged tree the aligner reads.

The full annotation therefore runs into `annotations_full/` — `annotate` hashes the
per-split manifests into its identity, so the re-split features need a fresh directory
(`annotations/` holds the first, leakage-failing attempt).

### Full forced alignment ✅ (8 min on the GPU)

`tiny_voca.annotate --features features2 --output annotations_full --dataset-revision
kss-24k-v1 --splits train validation test --config config-ko.json --device cuda
--max-weak-phone-fraction 0.6 --min-token-probability 0.3 --max-phone-frames 90`

| split | accepted | rejected | accepted hours |
| --- | --- | --- | --- |
| train | 8 504 | 3 706 | 8.448 |
| validation | 234 | 88 | 0.234 |
| test | 228 | 94 | 0.230 |

33 clips/s, 476 s for the whole corpus. Rejection reasons over train:
`overlong_phone` 2 000, `weak_alignment` 1 766, `low_confidence` 1 258,
`vowels_not_voiced` 134 — i.e. 70 % of the corpus, 8.4 h of clean Korean with
Jamo-level durations. 8.4 h is a comfortable corpus for a 4.5 M-parameter model, and
the extra 3.7 h of marginal alignments would mostly add label noise, so the QC limits
from the 20-clip gate are kept as they are.

### Training runs 5× faster than the probe

With the CPU free (the six `prepare` processes were what made the probe measure
2.9 steps/s), the full run settles at **14.0 steps/s, ~1 550 audio-s/s, batches of
112 audio-s (≈ 32 clips, 10 600 frames)** at 5.2 GB of VRAM. 104 000 steps is therefore
**~2.1 h**, not 9.6 h, so the target is extended to **200 000 steps (~4 h, 760 epochs
over 8.4 h of audio)** — `steps` is not part of the checkpoint recipe, so resuming with a
larger `--steps` is compatible.

`scripts/parse_train_log.py` was fixed for this: the driver log mixes UTF-8 markers with
UTF-16LE python output (decoded as UTF-8 with NULs dropped), `frames`/`utterances` are
per-log-window sums (so the rate is `frames / window-seconds`, not a cumulative
difference), and the table now shows the real batch size in audio seconds.

### GAN phase speed and the final step target

The adversarial phase costs ~4× the step time (critics + generator pass over the wave):

| phase | steps/s | audio-s/s |
| --- | --- | --- |
| teacher-forced (steps 1 – 8 000) | 14.0 | ~1 560 |
| GAN (from 8 000, full at 11 000) | 3.42 | ~375 |

104 000 steps therefore takes ~7.5 h, not 2.1 h. The target is set to **130 000 steps**
(≈ 9.6 h of GAN training, ends ~15:40, evaluation and samples by ~16:45, i.e. ~15.6 h
from the start of the session — inside the 18 h budget). Validation at 12 000 steps:
`predicted_mel 0.787`, `duration_length_ratio 1.063`, `pitch_error 2.96 st`,
`voicing_accuracy 0.929`, `pause_accuracy 0.946`.

## Phase 7 — full training ✅

`scripts/train_full.ps1 -Steps 130000` ran the 130 000-step recipe in three chunks
(4 h, 3.5 h, 1 h) resuming from `runs/full/last.pt`; total wall clock **05:52 → 17:27 =
11 h 35 min**. One driver bug was fixed on the way: `Current-Step` returned an array when
`torch` printed a warning to stdout, and PowerShell's `array -ge scalar` yields a
non-empty (always truthy) array, so the chunk loop stopped after the first chunk. It now
takes the last stdout line and casts to `[int]` (verified across four resumes:
54 996 → 95 063 → 122 000 → 130 000).

Final training record (step 130 000): `loss 9.99`, `mel 0.449`, `duration 0.062`,
`voicing 0.052`, `pitch 0.058`, `pause 0.002`, `stft 0.152`, teacher fraction 1.02
(GAN fully on), 3.24 steps/s, 358 audio-s/s. No NaNs in any of the 2 600 log records.

Validation (32 clips, on every 2 000-step save, 65 validations over the run):

| metric | step 12 000 | step 130 000 |
| --- | --- | --- |
| `predicted_mel` | 0.787 | 0.805 |
| `teacher_forced_mel` | – | 0.755 |
| `pitch_error_semitones` | 2.96 | 3.18 |
| `voicing_accuracy` | 0.929 | 0.926 |
| `pause_accuracy` | 0.946 | 0.948 |
| `duration_length_ratio` | 1.063 | 1.051 |

Checkpoints: `step-00008000.pt` (GAN start) + every 10 000 steps + `last.pt` +
`inference.pt` (17.4 MB, 4 549 579 parameters), 1.12 GB total.

## Phase 8 — samples, audio and ASR gates ✅

`tiny_voca.evaluate --checkpoint runs/full/inference.pt --splits test --per-voice 6
--novel 2 --device cuda` (16 s, 6 held-out clips synthesized free-running):

| metric | generated | reference |
| --- | --- | --- |
| CER | **0.1177** | 0.0256 |
| WER | 0.2083 | 0.0417 |
| duration ratio | 0.907 | – |
| pause ratio | 0.956 | – |
| mel DTW | 0.778 | – |
| pitch std (semitones) | 6.09 | 5.78 |
| voiced fraction | 0.673 | 0.572 |

`frontend_token_mismatches: []` — the front end reproduces the annotated tokens for every
evaluated sentence. Three of six test sentences score CER 0.0 %.

The 7 Phase 8 utterances (`samples/*.wav`, synthesized in 5.5 s) pass
`scripts/verify_wavs.py` (24 000 Hz, mono, 3.6–6.2 s, RMS 0.079–0.095, peak ≤ 0.627,
no NaN/Inf) and `scripts/asr_check.py` (faster-whisper `small`, `ko`, GPU):

| sample | CER | WER | CER (numbers folded) |
| --- | --- | --- | --- |
| 01 greeting | 0.0 % | 0.0 % | 0.0 % |
| 02 question | 0.0 % | 0.0 % | 0.0 % |
| 03 phonology | 10.0 % | 50.0 % | 10.0 % |
| 04 numbers | 54.5 % | 90.9 % | 12.5 % |
| 05 long | 15.6 % | 26.7 % | 6.5 % |
| 06 exclamation | 0.0 % | 0.0 % | 0.0 % |
| 07 code-mix | 30.0 % | 25.0 % | 30.0 % |
| **mean** | **15.7 %** | **27.5 %** | **8.4 %** |

`scripts/asr_check.py` gained the "numbers folded" column: Whisper writes Korean numerals
as Arabic digits, which a character-level score counts as several substitutions, so a
digit run and a run of Sino-Korean numeral morphemes are folded to one placeholder, with
the raw CER kept next to it. The code-mix clip is the honest weak spot: KSS has no English
audio, so the dizione phones are unseen at training time.

**Session total**: 01:10 → 17:40 ≈ **16.5 h** of the 18 h budget (prepare 2 h single +
22 min parallel, annotate 8 min, train 11 h 35 min, evaluate + samples + ASR ~10 min).

## Phase 9 — browser demo (Korean) — 16:20 → 17:10

Goal: port `tiny-voca/web` (upstream's English demo) to the Korean model, so the
130 000-step checkpoint runs in the browser and the repository publishes it.

### Export the graph — 21.4 MB, 40 s

```powershell
.\.venv\Scripts\python.exe -m tiny_voca.export --checkpoint runs\full\inference.pt `
  --output tiny-voca\web\public\assets\model.onnx            # 21.4 MB + model.json
.\.venv\Scripts\python.exe -m tiny_voca.export --checkpoint runs\full\inference.pt `
  --output runs\full\model-noise.onnx --external-noise        # for the exact check
```

`scripts/check_onnx.py` compares the graph with the torch model on 5 sentences
(including the code-mixed one). The shipped graph generates its own noise for the
source-filter head, so its waveform is a *different realisation of the same voice*:
durations must match exactly, the waveform only correlates.

| sentence | tokens | seconds | durations | correlation | peak |
| --- | --- | --- | --- | --- | --- |
| greeting | 47 | 4.11 | exact | 0.9937 | 0.138 |
| question | 44 | 3.67 | exact | 0.9824 | 0.163 |
| numbers | 61 | 4.70 | exact | 0.9876 | 0.218 |
| code-mix | 43 | 3.61 | exact | 0.9927 | 0.259 |
| exclamation | 63 | 5.97 | exact | 0.9938 | 0.153 |

**Gate**: durations exact on all 5, min correlation 0.9824, and the
`--external-noise` graph fed the torch noise stream reproduces the waveform to a
peak of **6.0e-4** (float32 conv accumulation, ≈ −64 dB). That second graph is what
proves the export is faithful rather than merely similar.

### Morphology in Pyodide — the one real porting problem

g2pK calls `python-mecab-ko` for POS tags; it is a native extension with no WASM
build. `scripts/mecab_shim.py` implements `.pos()` in pure Python (particle /
sentence-ending / bound-noun / suffix peeling — g2pK only reads
`tag.split("+")[-1][0]`, so the coarse class is all that matters). Tables alone
reach 88 %; the fix that mattered was the **corpus dictionary**:
`scripts/build_mecab_dict.py` records the real MeCab analysis of every eojeol in the
12 210 *training* sentences (20 649 eojeols, 0.95 MB), so the browser reproduces the
front end the model was trained on wherever the corpus has the word.

`scripts/check_shim.py --sentences 644` on the held-out splits (dictionary built
from train only, so this measures real coverage):

| measure | value |
| --- | --- |
| every eojeol in the dictionary | 26.2 % of sentences |
| **identical token stream** | **92.64 %** (428/462) |
| **token accuracy** | **99.80 %** (34 of 17 281 tokens) |
| mean edit | 0.07 tokens per sentence |

Residual diffs are one tensification or liaison token (`ᄀ`→`ᄁ`, `ᄉ`→`ᄊ`, `ᄄ`→`ᄃ`)
where the shim draws the stem boundary at a slightly different place than MeCab.
Gate set at 90 % identical; passes.

### Assets — 28.8 MB, `scripts/build_web_assets.py`

`model.onnx` 21.4 MB · `phonemizer/` 6.3 MB (dizione's built files + NLTK
`cmudict.zip`) · `mecab-dict.json` 0.95 MB · `pyfront.zip` (g2pK + jamo +
`kofront` = the *same* `vocab/text/korean.py` + the shim as `mecab.py`) ·
`dizione.zip`. They are committed to git so GitHub Pages needs no build step;
`tiny-voca/.gitignore` un-ignores `web/public/assets/` and `model.onnx`.

### Page

`web/public/index.html` (Korean: 6 example chips incl. 한영 혼용, 빠르기/음높이
sliders, 지표, 자모 보기) + a rewritten `worker.js` (Pyodide 0.26.4 + ONNX Runtime
Web 1.20.1, Cache Storage keyed on `VERSION`). The upstream English page stays as
`en.html` / `worker-en.js`.

### Measured in headless Chrome

| | local server | GitHub Pages |
| --- | --- | --- |
| audio | 3.6 s (code-mix) | 4.7 s (numbers) |
| synthesis wall clock | 0.37 s → 9.7× real time | 0.28 s → 16.9× real time |
| phonemization | 186 ms | 29 ms (warm) |
| page errors | none | none |

The code-mix sentence renders `ᅩ늘paɪθɑːnᅳ로…` — dizione for `Python`, Jamo for the
Korean around it, exactly as in training.

### Deploy

`.github/workflows/pages.yml` publishes `tiny-voca/web/public`. First run failed:
`configure-pages` could not create the site (free plan + private repo). Made the repo
public and added `enablement: true`; run 43852256917 succeeded →
**https://ooodev.github.io/korean-tts/**.

**Decision log**: two ONNX graphs instead of one (a single graph cannot show both that
durations are exact and that the waveform is reproduced); corpus dictionary instead of
a bigger rule table (rules plateau at ~90 %, the dictionary is 0.95 MB and exact where
it applies); assets in git instead of a Pages build step (a build would need PyTorch on
the runner for every docs-only commit).

---

## Phase 10 — Workspace cleanup (2026-11-17) ✅

User request: remove everything from the workspace that is safe to delete. Basis:
`.gitignore` + AGENTS.md both classify the intermediates as "not committed
(regenerable)"; every deleted path is gitignored, so `git status` stayed clean
throughout.

**Second pass (same day)**: user confirmed the source will be modified and the model
retrained from scratch, so *all* previously trained / completed model artifacts were
removed as well. This deleted committed files, so `git status` now shows 68 ` D`
entries (nothing else). Removed:

| Path | Content |
| --- | --- |
| `runs/` (whole) | `inference.pt`, `model-noise.onnx`, config/recipe/metrics — the final model |
| `annotations_full/` | trainer input manifests (re-run `tiny_voca.annotate`) |
| `samples/` | deliverable wavs + ASR report (re-run `synthesize_samples.py` + `asr_check.py`) |
| `eval/` | `eval/full` + `eval/precheck` reports & audio (re-run `tiny_voca.evaluate`) |
| `logs_train_full.txt` | the committed 130 k-step training log |
| `tiny-voca/web/public/assets/` | ONNX export of the old model for GitHub Pages (re-run `scripts/build_web_assets.py`) |

Side effects to remember after the retrain: the live Pages demo at
ooodev.github.io/korean-tts is served from deleted assets (restore with a new build +
commit), and `scripts/check_onnx.py --strict` needs `runs/full/model-noise.onnx`
re-exported. `weights/phonemizer/` was kept — it is the front-end dictionary, not a
trained model.

### Removed (24.45 GB; workspace went ~29.8 GB → 5.38 GB excl. `.git`)

| Category | Paths | Size |
| --- | --- | --- |
| Parquet shards (stage 1/2, regenerable via `scripts/prepare_kss.py` + `resplit_kss.py`) | `data/`, `data_parts/`, `data_fast/`, `data_smoke/`, `data_smoke100/` | 3.85 GB |
| Acoustic features (stage 3/4, regenerable via `tiny_voca.prepare` + `merge_features.py`) | `features/` (superseded 1st attempt), `features2/`, `features_w/`, `features_fast/`, `features_smoke/`, `features_smoke100/` | 17.86 GB |
| Smoke/probe alignments (regenerable via `tiny_voca.annotate`) | `annotations/`, `annotations_fast/`, `annotations_fast2/`, `annotations_smoke/`, `annotations_smoke100/` | 0.01 GB |
| Smoke/probe training runs | `runs/fast/`, `runs/fast2/`, `runs/full_probe/`, `runs/probe-smoke/`, `runs/probeA/`, `runs/probeB/`, `runs/probeC/`, `runs/smoke/` | 1.45 GB |
| Resume checkpoints of the full run (regenerable by retraining; only `inference.pt` + config/recipe/metrics are committed) | `runs/full/last.pt`, `runs/full/step-{00008000…00130000}.pt` | 1.25 GB |
| HF download cache (re-downloads on first use) | `weights/.cache/` | 0.00 GB |
| Smoke/probe eval output (regenerable via `tiny_voca.evaluate`) | `eval/evaluate_probe/`, `eval/fast/`, `eval/fast2/`, `eval/smoke/`, `eval/precheck_asr.json` | 0.01 GB |
| Per-stage logs (noise; decisions live in this file) | 18 × `logs_*.txt` except `logs_train_full.txt` | ~0.1 MB |
| Bytecode caches | `__pycache__/` under `tiny-voca/`, `scripts/`, `.github/` | ~0.05 GB |

### Kept (final state after the second pass)

Source only: `tiny-voca/` + `dizione/` + `scripts/` + docs + `config-ko.json` +
`.github/`. Plus `weights/phonemizer/` (front-end dictionary, not a trained model)
and `.venv/` (5.25 GB — rebuilding it costs more than it saves). Nothing trained or
exported remains; the workspace is a clean slate for the retrain.

### Verification

- `git status --short` → empty (no tracked file touched).
- Re-measured workspace: 5.38 GB excl. `.git`; `.venv` is 84 % of the remainder.
- Rebuilding the full pipeline from scratch costs stages 1–5 ≈ 45 min GPU/CPU
  (20+12 s+22+1.5+8 min) if the intermediates are ever needed again.

**Decision log**: deleted everything gitignored and documented as regenerable rather
than keeping "just in case" copies — the pipeline's reproducibility is its committed
artifacts + the scripts, and disk was the binding constraint from Phase 1. Kept
`model-noise.onnx` (20 MB) because `check_onnx.py --strict` defaults to it and
re-exporting via `scripts/build_web_assets.py --export` is a heavier gate than the
space is worth.

## 2026-11-17 — v2.0 bilingual spec locked (`V2_SPEC.md`)

**Decision log**: user approved a unified KO+EN 4.5M model (v2.0) after review of the
warm-start proposal. Locked decisions: D1 from-scratch training (v1.0
`runs/full/inference.pt` stays the baseline; warm-start fallback at the 120k gate
requires a snapshot-rebuild script since `last.pt` was deleted in the clean-slate pass);
D2 keep the 142-token inventory (EN IPA rows 0-75 and KR Jamo rows 76-141 are already
disjoint — zero phonetic collisions, 3-way stop contrast preserved; no redesign);
D3 Korean-accented English = loanword G2P to Jamo, not IPA substitution (reuses the
proven Korean acoustic pathway; IPA substitution produces a hybrid no model has heard);
D4 dual aligners for v2.0 (kalign covers loanword output since it carries Hangul
syllable units; unified CTC aligner deferred to v2.1 with a parity gate); D5 no language
embedding (disjoint token sets already encode language; add only if prosody bleed shows);
D6 corpus = KSS + n4ze3m/tiny-voca only (no synthetic code-switched data in v2.0).
Phase order: P0 git restore + dataset download, P1 front end (TN rules incl. fixes for
3개→삼개 and phone-number misreads, loanword G2P, ≥120-case gate), P2 EN alignment +
manifest merge, P3 160k-step train (~14 h GPU), P4 evaluation gates.

## 2026-11-18 — Phase 1 front end complete: TN engine, loanword wiring, 185-case gate green

**P1 status: 100% verified.** The `ko-loan` front end (Mode A) is built and the
golden-case gate passes. Order honored: P1 verified before any alignment/training.

### What landed (this session)

- `tiny-voca/src/tiny_voca/korean.py`
  - `korean_normalize(text)`: full Mode A TN, pass order
    etc → 3-part date → currency → % → time → phone → range → unit →
    decimal → a/b → ordinal → month → hour → counter → `X\d+` codes →
    ≥5-digit runs → bare digits (Sino) → symbol map. probe_tn 75/75.
  - Counter sets: native = g2pK BOUND_NOUNS + 명 장 팩 잔 끼 칸; Sino = 층 년 월
    일 분 초 원 회 등 점 호 기 차 페이지 코어 세 박 세기. Spacing: 세/박/살(>20)/
    번째 get a space; `0시` → `영 시`.
  - `loan_tokens()`: English span → `loanwords.loanword_span` (dict → acronym →
    IPA fallback), each syllable → NFD Jamo (silent initial dropped), `stress=0`,
    one labelled unit per syllable so kalign aligns loans like Korean words.
    Returns `unknown=[]` (IPA-fallback syllables are valid Hangul, never UNK).
  - `phonemize_mixed(text, frontend=None)`: `ko-loan` runs
    `korean_normalize(normalize_ko(text))` before `segment()`; `ko` byte-unchanged.
- `tiny-voca/src/tiny_voca/text.py`: `phonemize()` passes `frontend=_frontend`
  through to `phonemize_mixed`.
- `tiny-voca/src/tiny_voca/data/loanwords.tsv`: 447 entries; **line 11 fixed**
  `clock 크록` → `clock 클록` (correct standard loanspelling).
- `tiny-voca/tests/test_frontend_gate.py`: **185 golden cases** (≥ 120 required),
  one test method per row: TNTests 75 (korean_normalize exact), LoanwordTests 55
  (unit labels == dictionary loanspelling), SandhiTests 21 (== g2pK output),
  ModeASegTests 22 (exact label tuple, no `<unk>`, no digits), ModeBTests 12
  (`own` == dizione, save/restore).

### Gates hit

- `probe_tn.py` 75/75 (all V2_SPEC §2.2 rows).
- `probe_seg.py` 22/22 Mode A unit-label tuples.
- `test_frontend_gate.py` 185/185 (first run 184/185: one golden-case typo,
  `국회의원` expectation written `구회 의원` from a stale note — g2pK actually
  emits `구쾨의원`; fixed the case, not the engine, per the gate's own rule).
- Full suite: `Ran 252 tests … OK (skipped=2)`, exit 0 (67 baseline + 185 gate).

### Decisions

- **Golden-case errors are fixed in the data, engine errors in the code.** The
  gate docstring makes this explicit so a future "fix" that silently edits a
  reading leaves a trail.
- g2pK model quirks are pinned, not "corrected": 십이점→시비점, 삼점오→삼저모,
  십구세기→십꾸세기 are the same model v1 annotations used, so the aligner and
  the gate must agree with it.
- `10km를` bug class: `\b` treats Hangul as word chars, so unit boundaries need an
  ASCII-only lookahead `(?![A-Za-z0-9])`.

### Next

Phase 1 smoke stages (fresh prep + alignment smoke, fresh from-scratch training
smoke, inference/export/web-demo smoke Mode A+B), then Phase 2 ~18 h full
training on the joint corpus.

## 2026-11-18 — Phase 1 smoke stage in progress: pipeline rebuilt, EN alignment verified

### KSS data restored exactly (split-mismatch scare resolved)

`prepare_kss.py` (committed, unmodified) shuffles to 12 211/321/322, but v1's final
splits were 12 210/322/322 — because stage 2 `resplit_kss.py` re-groups the shuffled
rows sentence-wise (no transcript in two splits). Re-ran `resplit_kss.py --seed 42`
and verified the rebuilt `data/` shards against `annotations_full/` manifests:
**12 210 / 322 / 322, every id and every transcript matches** (probe
`probe_verify_shards.py`: PASS). KSS annotations reused as-is.

### EN corpus screen (`scripts/prepare_en.py`, new)

Reads the pinned snapshot from the HF cache (revision `63b6b896…`, refs/main), applies
the V2 §3 screen: Whisper WER ≤ 0.10, 0.5–15 s, < 512 tokens, zero `<unk>` under the
`own` front end. Writes KSS-compatible shards for `tiny_voca.prepare` (default Config:
speakers F2/M2, 24 kHz). Bug caught & fixed on first run: the shared-text check dropped
*within-split* duplicate takes (both voices read many sentences — valid rows; KSS has
them too); only cross-split text is dropped, and the smoke run counted **zero** of
those, so the upstream splits are sentence-unique. Smoke: 100 rows/split kept.

### EN alignment smoke (espeak-CTC path validated)

`tiny_voca.annotate --features features_en_smoke --output annotations_en_smoke
--limit 100` with `TINY_VOCA_FRONTEND=own`: **300/300 accepted, 0 rejected**, F2/M2
≈ 50/50 per split. Timing-probe stats match the shipped v4 annotations (vowel→voiceless
flip_found 0.89/0.80, fricative→vowel 0.99/0.94). Identity: acoustic
`a6ebc979…` (= KSS), frontend `c729f235…` (= v4 exactly), aligner
`facebook/wav2vec2-lv-60-espeak-cv-ft` rev `ae45363b…` (= v4 exactly).

### Merge/subset machinery (new)

- `scripts/merge_annotations.py`: `features_mixed/{ko,en}` are `mklink /J` junctions;
  rows get `ko/`/`en/` path prefixes; merged `annotations.json` declares
  `frontends`/`annotations` fingerprint sets + embeds both identities. Junctions
  repoint cleanly (smoke EN dir → full EN dir later).
- `scripts/subset_manifest.py`: reproducible smoke subset, stratified by speaker,
  verifies every selected `.pt` exists, carries the parent's declared sets + own
  fingerprint.
- `tiny_voca/data.py` patch (252 tests still OK):
  1. declared-set checks — metadata `frontends`/`annotations` dicts accept a row when
     its values are in the set; single-run manifests keep the strict checks.
  2. junction-safe containment — `Path.resolve()` follows junctions and broke the
     `is_relative_to` test; now a lexical check (`is_absolute() or ".." in parts`).

### Decisions

- **Training front end: `ko-loan`.** Checkpoint fingerprint matches Mode A exactly;
  Mode B (`own`) serves via `infer.load_model(check_frontend=False)` — the documented
  "same inventory, different conventions" escape hatch (EN ids 0–75 ⊂ 142 vocab).
- **Reuse `annotations_full` for KSS**, self-align EN with espeak-CTC per V2 §3 (model
  downloaded; revision matches v4's aligner exactly).
- EN `--config` omitted in prepare/annotate → `Config()` defaults (F2/M2, vocab 0 →
  76 under `own`) are the proven upstream settings; `config-v2.json` (KSS/F2/M2, 142)
  is for the KSS prepare and the merged training only.

### In flight / next

- KSS features: 6-process `prepare` + `merge_features.py` → `features2/` (background,
  ~22 min). Driver bug fixed: `split_shards.py` placeholder is `{index}`, not `{0}`.
- Then: merge → subset → 800-step smoke train (`runs/smoke_v2`, GAN 300→600, teacher
  released 200→400) → Mode A+B synthesis + ONNX parity → Phase 2 full.

### Verified: KSS is 100 % pure Korean (gate design consequence)

Probed all three `annotations_full` splits for ASCII in the transcript:
**train 0/8504, validation 0/234, test 0/228**. The `ko` vs `ko-loan` front-end
difference (Korean TN + loanword Jamo for English spans, see `korean.py
phonemize_mixed`) therefore never fires on corpus text: every KSS row's annotated
tokens are exactly what Mode A serving reproduces. The KO gate measured under
`ko-loan` is numerically comparable to v1's `ko`-gate (11.8 % CER), with no
code-mixing inflation. Code-mixing only appears in novel sentences and user text.

### Gate commands (V2 §5), designed

- **KO gate (Mode A)**: `TINY_VOCA_FRONTEND=ko-loan`,
  `evaluate --annotations annotations_mixed --splits test` — merged manifest carries
  the declared front-end sets, so KSS rows (`ko` fingerprint) load under `ko-loan`;
  metric = `summary.test.KSS.generated_cer` ≤ 12.5 %. (`annotations_full` alone would
  fail `data.py`'s strict single-front-end check under `ko-loan` — the merged manifest
  is the intended entry point.)
- **EN gate (Mode B)**: `TINY_VOCA_FRONTEND=own`,
  `evaluate --annotations annotations_en --splits test --asr-model small --asr-language en`
  (multilingual `small`; `small.en` is not in the HF cache) — metric =
  `summary.test.all.generated_wer` ≤ 1.5 %.

### `evaluate.py` v2 patches (suite re-run in flight)

1. `load_model(..., check_frontend=False)` — the bilingual checkpoint stores one front
   end (ko-loan); the EN gate run serves with the other (`own`) on the same inventory.
2. `korean = frontend() in ("ko", "ko-loan")` — ko-loan selects the Korean ASR.
(Per-row `tokens_match_annotation` left unconditional: it is only meaningful when the
gate runs against the matching corpus manifest, as designed above.)

### Smoke-merge plumbing: `--allow-incomplete`

`annotate --limit` bakes `limit` into the identity and writes `"complete": false`;
`merge_annotations.py` and `subset_manifest.py` now take `--allow-incomplete` for the
deliberate smoke merge (merged manifest stays `complete: false`; the subset manifest
is always `complete: true` — data.py requires it, and the subset verifies every `.pt`).

### Phase 2 in flight

- **KSS features** (background): 6 parallel `prepare` + merge → `features2/`.
- **EN full pipeline** (background, launched concurrently): `prepare_en.py` (full
  screen) → `data_en/` → split train into 6 → 8 parallel `prepare` → `features_en/`.
  CPU-bound like KSS; both jobs share the 24 threads.

### EN first wave died: 14-process storm, retry staggered

The full EN screen passed cleanly — **9,254 rows kept** (train 7,976 / val 702 /
test 576; drops were all `wer_too_high`: 250/16/20). But 7 of the 8 EN `prepare`
children died within minutes (only `validation` finished). Forensics:

- No native crash in the Windows Application log (no python.exe Application Error /
  BEX entries) → clean Python-level exit; the traceback went to the void window
  (`Start-Process` without redirection loses child output — the KSS driver has the
  same blindness, its log shows only the parent's lines).
- Each failed part dir held the 5 startup files (lock, manifest, rejected,
  preparation.json, features.json) but **0 clips and 0 rejections** → death between
  the metadata writes and the first clip; prime suspect is pyarrow mmap/commit
  failure under the 14-process storm (6 KSS + 8 EN, ~1.3 GB working set each plus
  torch VA), not per-row data.
- Side discovery: `.venv\Scripts\python.exe` is a shim that re-execs the uv base
  interpreter (`C:\Users\yuy\AppData\Roaming\uv\python\cpython-3.11-…\python.exe`);
  the parent stays idle (1 MB, 0 % CPU) — every "one" prepare job is really a
  parent+child pair, so the true process count was higher than the driver thought.

**Fix:** `build_features_en_full2.ps1` waits for `features2/` to finish (KSS job
first), then runs the 8 EN parts on a quiet CPU with
`Start-Process -RedirectStandardOutput/-RedirectStandardError` per child so any
error is captured (`logs_en_parts/<part>.err`), and dumps those tails on failure.
`train-0` resumes via `--resume` (187 clips journaled by a foreground probe).

### The storm also bit KSS: 9 MemoryError rejections, resumed clean

The first EN wave's 14-process storm was not the only victim: the KSS job finished
at **12 203 / 320 / 322** instead of 12 210 / 322 / 322. All 9 rejections were
`MemoryError: Unable to allocate …` — transient commit exhaustion, not data. Fix
(`resume_kss_parts.ps1`): strip the MemoryError records from the parts'
`*.rejected.jsonl` journals (so `--resume`'s `done` set releases the clips),
re-run `prepare --resume` **sequentially** (one child at a time on top of the 8
active EN children — 14 was the failure count), re-merge. All 9 clips accepted:
**features2 = 12 210 / 322 / 322, rejected 0**. Two driver bugs found on the way:
`Path.glob()` returns a generator in 3.11 (`+` between them raised TypeError,
silently skipping the cleanup), and the validation shard lives in `data/`, not
`data_parts/` (only train was split).

Final proof (`probe_features2_coverage.py`): for every split the id sets of
`annotations_full` (accepted + rejected) and `features2` match exactly, and every
manifest row's `.pt` file exists — **PASS**. `annotations_full` reuse is safe.

### Smoke train: output-dir collision fixed, re-running

`train.py` requires `--output` to be empty, but the subset driver had written
`runs/smoke_v2/manifest/` — inside the train output dir. Moved the subset to
`annotations_mixed_smoke/subset/` (beside the merged manifest it derives from);
merge/subset are idempotent and deterministic (seed 42), so the whole driver was
re-run. First merge output: **8,604 / 334 / 328** rows = KSS accepted
(8,504/234/228) + EN smoke (100×3).

### Smoke train ✅ — 800 steps, no NaN, both modes synth OK

60/20/20 subset (fingerprint `ff1c5180…`), 800 steps in **127.75 s**
(~10.5 steps/s pre-GAN, ~5.5 with GAN), base_loss 41.99 → 13.54, no NaN.
GAN engaged from step 350; teacher released 200→400 (fraction 1.0→0.0);
validation logged at 200/400/600/800 — at 800: predicted_mel 1.14,
pause_accuracy 0.931, duration_length_ratio 1.06, pitch_spread_ratio 0.95.
Checkpoint probe: `inference.pt` step 800, **4,550,347 params** (v1: 4,549,579 —
the +768 is the 2 extra speaker embeddings, 3 voices × 256-dim vs 1), all finite;
`last.pt` 21.7 M training-side params, all finite. Step checkpoints saved at
300/400/800.

Synthesis (`smoke_synth_v2.py`, Mode A `ko-loan`/KSS + Mode B `own`/F2,M2):
6/6 cases, peaks 0.42–0.61 (no clipping), durations 3.0–6.0 s, `verify_wavs.py`
all OK (24 kHz mono, RMS 0.079–0.105). Code-mixed case a3 (Python/USB/10km)
tokenized cleanly under `ko-loan`. (Artifacts are 800-step smoke output —
expected to sound rough; the gates below prove the machinery, not the voice.)

ONNX parity (`check_onnx.py` under `ko-loan`, both graphs 21.4 MB):
**durations exact on all 5 sentences** (incl. the code-mixed `Python` one),
min correlation **0.9922** (noise realisation differs, peak 0.126), strict
`--external-noise` graph fed the torch noise reproduces the waveform to a peak
of **8.7e-4** (≈ −61 dB) → **OK**.

**Smoke stage complete** — every v2 stage (prep, alignment, merge, train, synth
Mode A+B, ONNX export/parity) has run end-to-end on the mixed manifest.

### EN gate: `--serve-frontend` — the bilingual-checkpoint serving constraint

Dry-run of the EN gate (Mode B, `own` inventory) against the smoke checkpoint
exposed the constraint the patch is for: the v2 checkpoint embeds a 142-token
config, and `Config.__post_init__` validates `vocab_size == len(SYMBOLS)` against
the **active** front end — so the checkpoint can only *load* under `ko`/`ko-loan`
(142), never under `own` (76). Fix in `evaluate.py`: new `--serve-frontend`
option — load under the training front end, then `set_frontend()` after data
loading and before synthesis. Sound because `own`'s 76 tokens are the exact
prefix of `UNIFIED_VOCAB` (vocab.py: "every id produced by dizione.phonemize is
valid in both"). The EN gate therefore runs `TINY_VOCA_FRONTEND=ko-loan
evaluate --serve-frontend own` on a **merged-manifest** slice (`gate_manifest.py`
filters F2/M2 rows from `annotations_mixed`; single-run `annotations_en` would
fail data.py's strict check under the ko-loan process). Verified on the smoke
checkpoint: 100 EN rows, **`frontend_token_mismatches: 0`** — the serve switch
reproduces the espeak-CTC tokens exactly.

Gate runners: `scripts/run_v2_gates.ps1` (KO: KSS-filtered merged manifest,
`--per-voice 300 --novel 0`, metric `summary.test.KSS.generated_cer` ≤ 0.125;
EN: F2/M2-filtered merged manifest + serve-frontend, `--per-voice 600`,
`summary.test.all.generated_wer` ≤ 0.015; writes `verdict.json`). `gate_watcher_v2.ps1`
parks alongside training, waits for `runs/v2/inference.pt` to reach each
decision step (rewritten atomically every 2000 steps), pins a copy, gates it —
with a CPU retry if the CUDA gate fails (VRAM headroom check).

### `Start-Process` exit codes are broken in this environment

The EN wave-2 driver reported 8/8 failures — two children had actually finished
all 1330 clips. Experiment: `sys.exit(3)` via `Start-Process -PassThru` reports
**ExitCode 1** (even the raw uv interpreter; even with `-Wait`); with
`-RedirectStandard*` it is **null**. Foreground `& exe; $LASTEXITCODE` is exact
(7, 9, 3 all propagated) — including through the venv shim. So: child processes
are only launchable via Start-Process (for redirection) when success is verified
by **output state**, never by exit code. Wave-3 driver verifies each `prepare`
child by the JSON summary line it writes to the captured stdout, and does all
merge steps in the foreground.

### EN wave-2 driver bugs (captured for the record)

1. Malformed condition `$p -ne "train-1" -ne "train-2" …` — chained `-ne` without
   operands parses as `$p -ne "train-1" -ne "train-2"` = `($p -ne "train-1") -ne
   "train-2"`, a bool-vs-string comparison whose result is unpredictable
   (train-2…5 got `--split train-2` etc.; train-0/1/validation/test happened to
   pass the broken test by accident).
2. `validation`/`test` parts referenced `data_parts_en\*.parquet` — only the
   train shard was split; those two are single shards in `data_en/`.
3. The `$p: $line` string — `$p:` parses as a drive-qualified variable reference
   (InvalidVariableReferenceWithDrive); the script died at parse, nothing ran.

Wave 3 (running): the remaining 6 parts in two waves of three (3 prepares ≈ 12
real processes incl. shim re-exec — under the 14-process storm threshold),
sentinel-verified.

### EN features complete — Phase 2 training launched

Wave 3 finished all 6 remaining parts first-try, **0 rejections**: features_en =
**7,976 / 702 / 576** (4.87 GB, every manifest row has its .pt). Three-at-a-time
confirm the storm threshold: the 14-process first wave lost 7 children, 6×
single-process sequential resume lost nothing.

Launched:
- **Main chain** `run_v2_full.ps1`: full EN annotate (espeak-CTC, GPU, ~15 min)
  → merge (re-points `features_mixed/en` → `features_en`) →
  `train_full.ps1 -Steps 160000 -Frontend ko-loan -Config config-v2.json`
  (~14 h at v1's ~3.1 steps/s, 4-hour resumable chunks).
- **Gate watcher** `gate_watcher_v2.ps1`: pins `runs/v2/inference.pt` at the
  80k and 120k decision steps (rewritten atomically every 2000 steps) and runs
  the bilingual gates on it; CPU retry if the CUDA gate fails (VRAM headroom
  against the live training job).

### D1 warm-start contingency staged

`scripts/build_warm_snapshot.py` (new) rebuilds the training-format snapshot
V2_SPEC D1 needs from v1's slim `runs/full/inference.pt`: `--init-from` reads
`system`/`critics`/`optimizer`/`critic_optimizer`/`recipe`, and its strict
config check rejects v1's KSS-only config — so the snapshot embeds config-v2,
v1 generator weights under the `model.` prefix (as-is; `load_compatible`
places them at branch start), fresh critics + critic moments (the branch logic
takes trained critics over when step > gan_start, so these fresh v2 ones are
what the warm run inherits — GAN scaling tolerates it), step 130000, v1
frontend/annotation fingerprints (started with `--new-data`). Verified against
the real v1 checkpoint: **182 exact, 2 grown** (`model.voices.weight` +
`model.frame_voices.weight` (1,192)→(3,192) — the +768 params over v1),
`mel_head` new, 0 dropped → `eval/v2/warm-snapshot.pt` ready. The warm run is
then `train.py --init-from eval/v2/warm-snapshot.pt --new-data --output
runs/v2-warm …` with the train_full recipe, 130k→160k.

### Gate machinery built and dry-run verified

`scripts/run_v2_gates.ps1`: bilingual gates for one inference checkpoint.
KO gate = KSS rows only (`gate_manifest.py` filters the merged manifest's
test split, keeping the parent's declared front-end/annotation sets so
data.py accepts the rows; `complete: True` forced — every row is a verified
copy), served under `ko-loan` with Korean ASR, `--per-voice 300 --novel 0`
(228 KSS rows → all), metric `summary.test.KSS.generated_cer` ≤ 0.125.
EN gate = F2+M2 rows, same mechanism, served under `own` with English ASR
(`--serve-frontend own` — see below), `--per-voice 600` (576 rows → all),
metric `summary.test.all.generated_wer` ≤ 0.015. Writes `verdict.json`.

**EN gate constraint discovered (dry-run):** the v2 checkpoint embeds a
142-token config; `Config.__post_init__` validates `vocab_size` against the
**active** front end, so it only *loads* under `ko`/`ko-loan`, never `own`
(76). Fix in `evaluate.py`: new `--serve-frontend` option — load under the
training front end, then `set_frontend()` after data loading, before
synthesis. Sound because `own`'s 76 tokens are the exact ID-prefix of
`UNIFIED_VOCAB` (vocab.py: "every id produced by dizione.phonemize is valid
in both"). EN gate therefore runs `TINY_VOCA_FRONTEND=ko-loan
evaluate --serve-frontend own` on a **merged-manifest** slice (single-run
`annotations_en` would fail data.py's strict check under ko-loan). Dry run on
the smoke checkpoint: 100 EN rows, `frontend_token_mismatches: 0` — the serve
switch reproduces the espeak-CTC tokens exactly.

`gate_watcher_v2.ps1`: runs alongside training, waits for
`runs/v2/inference.pt` to reach each decision step (80k/120k; the file is
rewritten atomically every 2000 steps, so a copy pinned then is a complete
checkpoint), pins it, runs the gates — CUDA, with a CPU retry if the gate
fails (VRAM headroom against the live training job).

### EN annotate + merge complete, full training started (12:01)

Full EN annotate (espeak-CTC, `own` front end): **complete in 3.3 min** —
train 7,947/7,976 (29 rejected: vowels_not_voiced 24, weak_alignment 4,
low_confidence 2), validation 699/702, test 574/576. Per-voice timing probe
healthy (within-2-frames ≥ 0.81 on both boundary classes, both voices).
Merge (re-pointed `features_mixed/en` → `features_en`): `annotations_mixed`
fingerprint `3b930c46d7498d22` → **train 16,451** (KSS 8,504 + EN 7,947) /
**validation 933** (234 + 699) / **test 802** (228 + 574).

Training (`train_full.ps1 -Steps 160000 -Output runs\v2 -Frontend ko-loan`,
4-hour resumable chunks): at step 4,650, 13.97 steps/s, loss 9.22 — v1's log
shows the identical trajectory (14.10 s/s at step 5k, dropping to ~3.2 s/s
after GAN starts at 8k). ETA: **80k ~18:30, 120k ~22:00, 160k ~01:15 (Nov 19)**.

### Pre-gate staging (12:30)

- **EN gate ASR switched to `small.en`** — V2_SPEC §5 names small.en explicitly;
  downloaded + load-tested now (was `small --asr-language en` to dodge a
  mid-run download).
- **`scripts/battery_v2.py`** (new) — the §5 listening batteries, imported from
  the P1 golden tables so the expected readings are the locked contract:
  number battery 20 cases (counters/phones/dates/percent) + loanword battery
  20 words, Mode A KSS, text-level pre-check (must match 182/182 discipline)
  then synthesis + Whisper-ko CER as an automatic proxy for listening.
- **`scripts/listening_v2.py`** (new) — 16-case Mode A+B listening set for the
  final checkpoint (greetings, numbers, code-mix, flow, questions, stress).
- **20k trend probe job launched** (spec P3 "checkpoint probes at 20k/80k/120k";
  80k/120k are the gate watcher's): pins `inference.pt` at ≥ 20000 and speaks
  one Korean (KSS, ko-loan) + one English (F2, own) sentence to
  `eval/v2/probe-20000/`.
- Q3 (160k vs 190k step budget) is decided at the 120k gate from CER/WER trends.
- P5 (browser demo) scope checked: `worker.js` needs mode A/B phonemisation
  routing (kofront exposes `phonemize_mixed(text, mode)`), voice selection
  KSS/F2/M2, re-export of model.onnx under `ko-loan`, `VERSION` bump; parity via
  `check_onnx.py` + `check_shim.py`. Starts after the final gate passes.

## 2026-11-18 ≈14:30 — P5 demo code complete + `--serve-frontend` regression tests (training: step 18 000)

### Browser demo (P5) code complete — validation at final gate

- `worker.js`: speak messages now carry `voice?`/`mode?`; voice→mode per
  V2_SPEC §0 (KSS → `ko-loan` Mode A, F2/M2 → `own` Mode B, explicit `mode`
  overrides). The Pyodide phonemiser now calls `ktext.set_frontend(mode)` +
  `ktext.phonemize(text)` — the single faithful dispatch (`ko`/`ko-loan` →
  `phonemize_mixed`, `own` → dizione path), exactly the path the annotations
  used. Deliberately **not** `phonemize_mixed(text, "own")` directly: it differs
  from the standard path on punctuation sequences (quote+comma inserts other
  spacing — 4 of 10 EN corpus sentences differ), which would break the EN gate's
  `frontend_token_mismatches: 0` property. `synthesize` now takes the voice
  index (position in `config.voices`); `fit` threads the mode through its
  recursion.
- `index.html`: voice `<select>` populated from the worker's `ready.voices`;
  examples carry their voice (6 KSS Mode A + F2/M2 Mode B sentences); bilingual
  header/footer copy; facts grid carries the v2 structure (4,550,347 params,
  3 voices, 142 tokens, 160 000 steps) with `<!-- PUBLISH -->` markers for the
  ONNX size, elapsed training time, and the final gate numbers.
- Validated so far: `node --check` on both scripts (node v23); the worker's
  exact runPython block replayed in the venv — 11/11 checks passed (both modes
  phonemise, repeated `set_frontend` switches stable, the `own` path
  byte-identical to the annotations' `phonemize()` under `own`, ids inside the
  142-token vocab, `own` ids < 76). `VERSION` still "1" and no assets changed
  yet — publish step (after the final gate): re-export `model.onnx` under
  `TINY_VOCA_FRONTEND=ko-loan`, bump VERSION, `check_onnx.py`, `check_shim.py`.

### Regression tests for the `--serve-frontend` mechanism

- `tiny-voca/tests/test_frontend_serve.py` (new, 5 tests): (1) the motivating
  failure — `Config(vocab_size=142)` only builds while a 142-token front end is
  active; (2) a bilingual checkpoint loads only under the training front end
  (`check_frontend=False` doesn't help: `Config.__post_init__` rejects the
  vocab before the fingerprint is ever compared); (3) the gate order — load
  under `ko-loan`, `set_frontend("own")`, `speak()` gives finite audio with
  ids < 76 only, F2's voice index = 1, and the switch reverses back to Korean;
  (4) the two modes phonemise the two corpora into disjoint id ranges (the
  worker's routing property); (5) ASR defaults — `ko`/`ko-loan` → (small, ko),
  `own`/`espeak` → (small.en, en).
- `evaluate.py`: extracted `asr_defaults(frontend_name)` (behaviour identical;
  the inline logic was untestable).
- Suite: **257 OK, 2 skipped** (252 baseline + 5 new).

### Latent test-suite quirk discovered (documented, not fixed)

- `TinyVoca(config=Config())` freezes its default config at **import time**,
  and `test_ko_en_frontend.py` does
  `os.environ.setdefault("TINY_VOCA_FRONTEND", "ko")` at import and restores the
  frontend from that same variable in `tearDownModule` — so running the suite
  **without** the variable leaks the `ko` front end into `test_model`, where
  `test_default_model_is_under_five_million_parameters` compares the
  call-time `Config()` (142 tokens → 4 549 963) against `TinyVoca()`'s
  import-time frozen config (76 tokens → 4 537 291) → failure.
- The documented invocation (`$env:TINY_VOCA_FRONTEND="ko"` set first) makes
  both sides 142, which is why the suite is green (257 OK). If variable-less
  invocations ever become normal, the one-line hardening is `TinyVoca(Config())`
  in that test.

### Training progress

- Step 18 000; post-GAN **3.73 steps/s** (faster than v1's 3.2 at the same
  stage); ETA for 160k ≈ 01:10 Nov 19, in line with the original forecast.
- The 20k trend probe should fire within minutes; 80k gate watcher parked.

### 20k trend probe — green (~14:42)

`inference.pt` pinned at step 20 000 (18.3 MB); both serving modes synthesised
finite, healthy audio at 12.5 % of training: `ko.wav` 6.46 s (peak 0.65, rms
0.07, KSS/ko-loan) and `en.wav` 3.24 s (peak 0.84, rms 0.13, F2/own). The
Mode A + Mode B inference paths the final gates use are sound.

Also this window: `scripts/publish_v2_demo.ps1` (new, parse-verified) automates
the post-gate publish steps (mecab dict → assets+export → strict export →
VERSION bump → `check_onnx.py` → `check_shim.py`); `FINAL_REPORT_v2.md`
scaffolded with PUBLISH markers; verified work committed as
`c88719a` (28 files: ko-loan front end, EN data pipeline, gate machinery, P5
demo code, docs).

## 2026-11-18 ≈13:45 — Gate watcher incident: `run_v2_gates.ps1` parse error (fixed, gates re-running)

### What happened

The gate watcher pinned `inference.pt` at step 80 000 and step 120 000
(correctly, before each gate run) but the gate itself never executed:
`powershell -File scripts\run_v2_gates.ps1` died with a ParserError
(`InvalidLeftHandSide` at the param block). Both the CUDA run and the CPU
fallback failed identically; the watcher logged `GATE_WATCHER_DONE`.

### Root cause

`run_v2_gates.ps1` opened with a Python-style docstring (`"""…"""`,
lines 1-13). PowerShell 5.1 parses the first `""` as an empty string, the
third `"` as the start of a multi-line string, and the param block's token
state is corrupted by the time it is reached. The parser API confirms the
errors at 15:27…20:25. The earlier "dry-run verified" note covered the gate
logic (manifests, `--serve-frontend` evaluate, verdict math) but not the
PowerShell parse — lesson: **parse-verify every .ps1 with the Parser API**
(done for `publish_v2_demo.ps1` and now for this one; `train_full.ps1` was
clean).

### Fix + recovery

- Docstring → `#` comment block; `run_v2_gates.ps1` now parses clean.
- The pinned checkpoints are the decision artifacts (slim format,
  18 262 661 B each): `eval/v2/gate-80000/inference.pt`,
  `eval/v2/gate-120000/inference.pt`. Training kept running through the
  incident (step ≈ 121 000 at 13:45, ~3.4 steps/s).
- Re-running both gates on the pinned checkpoints, 120k first (D1 decision:
  KO CER ≤ 12.5 %, EN WER ≤ 1.5 %), then 80k (reference trend) — background
  job, `logs_v2_gate_rerun.txt`. Decision on completion: KO CER > 12.5 % at
  120k → D1 warm-start branch (stop chain, `--init-from eval/v2/warm-snapshot.pt`);
  otherwise continue to 160k and decide Q3 (160k vs 190k) from the
  80k→120k CER/WER trend.

### Disk cleanup (user request — 20 GB free at training ~step 9k)

Free space 20.3 → **36.3 GB** (+16 GB). Deleted, all consumed/re-derivable and
verified unreferenced by the live training (manifest rows point at
`ko/…`/`en/…` through the `features_mixed` junctions):

- `features_w_en/` 4.87 GB — EN feature shards, merged into `features_en`
- HF `datasets--Bingsu--KSS_Dataset` 3.63 GB — KSS audio, consumed by `data/`→`features2`
- HF `models--kresnik--wav2vec2-large-xlsr-korean` 3.59 GB — KO aligner, annotation done
  (v2 reuses `annotations_full`)
- HF `datasets--n4ze3m--tiny-voca` 1.95 GB — EN audio, consumed by `data_en/`→`features_en`
- `data_parts_en/` 0.94 GB — EN train shards, consumed by `features_en`
- HF `models--facebook--wav2vec2-lv-60-espeak-cv-ft` 1.20 GB — EN aligner, annotation done
- `runs/smoke_v2/` 0.37 GB — smoke checkpoints, stage complete (retrainable in 128 s)

Kept: `features2/` + `features_en/` (live reads), `runs/v2/`, `runs/full/`
(v1 baseline + D1 source), `data/` + `data_en/` (canonical re-prep input), all
non-smoke `annotations_*`, Whisper small + small.en (gate ASR), `.venv`.
If alignment is ever re-run, the two CTC aligners re-download (~4.8 GB).

- Smoke train driver updated for `--allow-incomplete`; `train_full.ps1` carries
  `-Frontend` (default `ko` preserves v1); recipe unchanged from v1 (batch 12000/48,
  warmup 1000, half-life 100 000, GAN 8000/3000/0.7, teacher 3000/7000/0).

## 2026-11-18 ≈14:20 — Gate re-run verdicts: both PASS. D1 closed, Q3 = 160k

### Verdicts (`eval/v2/gate-{80,120}000/verdict.json`, pinned checkpoints)

| Step | KO CER | pass (≤ 12.5 %) | EN WER | pass (≤ 1.5 %) | KO samples | EN samples | mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80 000 | 0.0601 | ✅ | 0.009 | ✅ | 228 | 574 | 0 |
| 120 000 | 0.0567 | ✅ | 0.0079 | ✅ | 228 | 574 | 0 |

Trend 80k→120k: KO −0.34 pt, EN −0.11 pt — both still improving, both far under
the bars (KO at 5.67 % vs v1's final 11.8 %).

### Decisions

- **D1 (120k decision gate): PASS — no warm start.** The staged
  `eval/v2/warm-snapshot.pt` remains untouched; training continues from step 0
  in `runs/v2`.
- **Q3 (step budget): 160 000.** Both metrics pass with wide margin at 120k and
  are still improving; the reserved 190k extension was for borderline outcomes,
  so the original budget stands.

### State

- Main chain alive at step ≈ 127 000, 3.60 steps/s → 160k ETA ≈ 16:58 local.
- Final gate after the chain: `scripts/run_v2_gates.ps1 -Checkpoint
  runs\v2\inference.pt -Tag final -Device cuda`, then
  `battery_v2.py` + `listening_v2.py`, then P5 publish.
- Report updated: `FINAL_REPORT_v2.md` now carries the decision-gate table and
  Q3 outcome (final-gate table still a PUBLISH marker).

## 2026-11-18 ≈17:25 — Training 160k DONE, final gate PASS, P5 publish green

### Training (Phase 3 complete)

- `runs/v2`: **step 160 000, finished 16:59**, `=== training finished at step
  160000 ===`, **0 NaN**, 4 resumable 4-h chunks (0 → 54 705 → 106 008 →
  155 656 → 160 000), **0 chunk failures**, 0 chunk retries.
- Wall clock **12 h 22 m** (44 415 s chunk time + restarts); post-GAN
  throughput ≈ 3.5 steps/s (v1: 3.2); final validation: duration ratio 1.003,
  pause accuracy 0.997, pitch spread ratio 0.99, pitch error 3.66 st.

### Final gate (`eval/v2/final/verdict.json`) — PASS

| Gate | Samples | Metric | Bar | Result |
| --- | --- | --- | --- | --- |
| KO (KSS, Mode A) | 228 | generated CER | ≤ 12.5 % | **5.35 %** ✅ |
| EN (F2/M2, Mode B) | 574 | generated WER | ≤ 1.5 % | **0.80 %** ✅ |

Per-voice: F2 WER 0.67 % (284), M2 WER 0.92 % (290).
`frontend_token_mismatches: 0` both gates. Full metric set in
`FINAL_REPORT_v2.md` §6. Honest notes: pause ratio ~1.1–1.2; F2 pitch spread
0.89× reference (M2 1.01×) — outside pass bars, flagged for the listening pass.
Trend: 80k 6.01 % → 120k 5.67 % → **final 5.35 %** (still improving at the end);
EN 0.90 % → 0.79 % → 0.80 % (flat at the bottom).

### Batteries + listening set

- Battery (`samples_v2_battery/battery_summary.json`): text level **39/39**
  (19 number + 20 loanword, P1 golden contract intact); Whisper-small-ko ASR
  proxy mean CER **58.9 %** — noisy proxy, not the spec gate: digit-preference
  bias (`세명` spoken → `3명` transcribed, `10km` → `10km`) + short-word
  instability (`AI` → `예, 예, 예`, `bag` → `big`). 9/20 loanwords exact.
- Two `battery_v2.py` bugs fixed this window: `units_of` now reads
  `(label, start, end)` triples of `Phonemized.units` (old code joined tuples
  → TypeError); case names now deterministic (blake2b) + filesystem-safe (the
  input `3/14` produced `...\3\14.wav` → LibsndfileError). Verified:
  `hello` → `헬로`.
- Listening set (`samples_v2/`): 16/16 wavs (8 Mode A KSS + 8 Mode B F2/M2),
  `verify_wavs` all OK, 24 kHz mono, peaks 0.44–0.98.

### P5 publish (`logs_v2_publish.txt`) — both demo gates green

- MeCab dict: 20 649 eojeols from the KSS train split, 0.95 MB, 100.00 %
  eojeol coverage.
- `model.onnx` 21.44 MB (re-export under ko-loan), first-visit assets 28.8 MB;
  **VERSION 1 → 2**.
- `check_onnx.py`: durations exact, min corr **0.9826** (noise realisation
  differs, peak 0.199), strict `--external-noise` graph peak **0.000906**
  (~−61 dB, float32 conv accumulation) → OK.
- `check_shim.py`: 462 held-out sentences / 17 281 tokens, **92.64 %**
  identical (gate ≥ 90 %), token accuracy **99.80 %** (34 tokens differ).

### Incidents / observations

- The harness delivered **premature "completed" notifications twice** (final
  gate, publish) while the child python was still running (seen as an
  "orphan" GPU process mid-EN-ASR). Lesson: trust the sentinel artifacts
  (verdict.json / final log lines), re-verify on the second notification.
- Publish pipeline reported exit code 1 despite all gates green: PS 5.1
  artifact — native stderr (export DeprecationWarning) piped through the outer
  `2>&1 | Tee-Object`; not a script defect (gates print their own OK lines).
- README typo fixed: `双语` → `이중언어` (stray CJK in a PowerShell comment).

### Next

Push → Pages build → published-site measurement (fill the last PUBLISH
markers in README/GUIDE/FINAL_REPORT) → final commit.

## 2026-11-18 ≈18:05 — Incident: published page dead (worker.js encoding corruption)

The post-push Pages verification failed: headless Chrome measured the published
site, but the page never reached `준비 완료` — `#status` stuck at
`작동 중: Uncaught SyntaxError: Invalid or unexpected token` (a probe that logs
page/worker console showed the error at +0 s, before any asset download).

**Root cause.** `scripts/publish_v2_demo.ps1` step 4/6 bumped VERSION in
`worker.js` with `Get-Content -Raw` / `WriteAllText`. PS 5.1 reads a BOM-less
UTF-8 file with the ANSI codepage (CP949 on this machine), so every Korean
status string in `worker.js` (11 lines) came back mojibake, and a byte sequence
that maps to nothing in CP949 became `?` — the corrupted `?` then unbalanced a
string literal → SyntaxError. The publish run's own stderr (export
DeprecationWarning) had already shown the pipeline was shuffling the file
through native encodings; the corruption shipped in `843d62f` and stayed on the
site for ~30 min until the measurement caught it.

**Fix (commit `a07e2dc`).**
- Restored `worker.js` from `c88719a` (the last intact version), re-applied the
  VERSION 1 → 2 bump with Python (byte-exact), verified the diff against
  `c88719a` is exactly one line, and node-`vm` syntax-checked the result.
- `publish_v2_demo.ps1` now reads with
  `[System.IO.File]::ReadAllText(path, [System.Text.Encoding]::UTF8)` and keeps
  the BOM-less UTF-8 write; the exact bug is noted in a comment.
- Deployed build verified over HTTP: VERSION 2, zero mojibake fingerprints,
  Korean status strings intact.

**Bonus damage found while scanning: GUIDE.md.** The fingerprint scan for the
worker bug surfaced ~35 corrupted lines in `GUIDE.md` — pre-existing since
`d3c077d` ("Document the browser demo across the docs", the v1-era commit that
rewrote the file through the same PS 5.1 ANSI pipe). It had mangled every
Korean sentence in the deliverables tree and ASR table, the `--text` example,
the phonology notes, and arrow/dash glyphs (`←` `→` `—` `…` `–` `≈` `×` →
`?`/`??`), plus a BOM at byte 0. Repaired all ~50 lines from intact sources:
`e624b81:GUIDE.md` (pre-corruption), `samples/manifest.json` (the 7 sample
texts), `scripts/make_guide.py` (same content as `guide.html`), and the
`README.md`/`guide.html` copy of the `--text` sentence. Verified: zero
fingerprint hits remain, every repaired line read back clean.

### Second incident: pyfront.zip missing front-end data (code-mixed chip crash, then OOV fallback)

Re-measuring the encoding-fixed site still failed, now on chip 2 — the
worker's catch surfaced `ModuleNotFoundError: No module named
'kofront.loanwords'` on `오늘 Python으로 테스트를 진행했습니다.`
`build_web_assets.py` shipped only `vocab/text/korean.py` as `kofront/`,
while `korean.py:438` lazily imports `loanword_span` from `kofront.loanwords`
for every English span inside Korean text. Pure-Korean input never takes
that path, so nothing caught it. Fix: add `loanwords.py` to
`FRONTEND_FILES`, rebuild, and verify the shipped zip end-to-end by
extracting it and phonemizing all three chip sentences (both modes) from the
extracted tree only. Commit `ddfb628`, VERSION 3.

The per-chip probe then exposed a subtler gap: the browser rendered `Python`
as `바ᅵ산ᅳ로` (OOV IPA→jamo fallback) while the trained front end renders
the dictionary spelling `파ᅵ썬ᅳ로` (파이썬; `python\t파이썬` in
`data/loanwords.tsv`). `loanwords.py` resolves its dictionary relative to
`__file__` (`kofront/data/loanwords.tsv`), which the zip never shipped — so
every code-mixed English word in the browser fell back to the OOV engine,
off the training distribution. Fix: ship `kofront/data/loanwords.tsv`
(462 entries); verified the extracted zip now reproduces the trained
rendering byte-for-byte on the chip sentences. Commit `32805ef`, VERSION 4.

Lesson recorded in the build script: the demo gates (`check_onnx`,
`check_shim`) run the venv's `tiny_voca`, which silently includes both
files; the shipped zip must be exercised end-to-end, not just the local
package.

## 2026-11-18 ≈20:00 — Published site verified; PUBLISH markers filled

VERSION 4 deploy confirmed (deployed `pyfront.zip` SHA256 == local).
Headless-Chrome measurement, 0 page errors, first-visit load 6.5 s,
READY "준비 완료 — 스텝 160000, 음성 KSS · F2 · M2.":

| chip | voice / mode | audio | synth | real time | phonemization |
| --- | --- | --- | --- | --- | --- |
| 숫자 | KSS / A | 4.4 s | 0.28 s | 15.7× | 33 ms |
| 한영 혼용 | KSS / A | 3.5 s | 0.21 s | 16.2× | 64 ms |
| 영어 | F2 / B | 4.1 s | 0.35 s | 11.9× | 149 ms |

Mixed chip renders `Python` as `파ᅵ썬ᅳ로` (파이썬, the dictionary spelling) —
the `loanwords.tsv` fix is live. Filled the last PUBLISH markers in
`tiny-voca/web/README.md` (실측 table), `GUIDE.md` (Measured-in-Chrome row) and
`FINAL_REPORT_v2.md` (Browser-demo row + §8 Published-site block); the v1
numbers stay as one historical line in the README. Final commit closes v2.0.

## 2026-11-18 ≈20:30 — .gitignore: inline comment broke features_mixed/

`git status` kept showing `?? features_mixed/` even though the directory was
listed for ignoring. Root cause: the entry was written with an inline comment
(`features_mixed/     # junctions into features2/ + features_en/`) and git
ignore patterns have no inline `#` — the whole line, comment included, became
the pattern and matched nothing. Fix: comment moved to its own line; the
v2 feature dirs are now in AGENTS.md's not-committed list too.

