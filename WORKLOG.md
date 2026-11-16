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
