# DATASET.md — KSS → Parquet → features → alignments

Four layers, each a directory with a JSONL manifest. Every layer is regenerable and every
stage journals its progress so it can be resumed.

```
Bingsu/KSS_Dataset (HF, 44.1 kHz)
  └─ scripts/prepare_kss.py ────────────► data/<split>-00000-of-00001.parquet
                                            └─ python -m tiny_voca.prepare ──► features2/
                                                                                 └─ tiny_voca.annotate ─► annotations_full/
```

## 1. `data/` — 24 kHz Parquet

`scripts/prepare_kss.py` loads `datasets.load_dataset("Bingsu/KSS_Dataset", split="train")`
(12 854 rows), resamples 44 100 → **24 000 Hz mono**, and keeps `expanded_script` as `text`
(numbers and units are already written as spoken Korean, punctuation preserved).

Filters: `0.5 s ≤ duration ≤ 15 s`, token count `< 512`, audio `< 32 s`, non-empty
transcript, non-silent. `voice_id = "KSS"`, ids `kss-00000 …`.

| split | rows | hours | distinct sentences |
| --- | --- | --- | --- |
| train | 12 210 | 12.211 | 11 833 |
| validation | 322 | 0.326 | 313 |
| test | 322 | 0.321 | 317 |
| **total** | **12 854** | **12.858** | 12 463 |

Schema (verified against `tiny_voca.prepare`):

| column | type | note |
| --- | --- | --- |
| `audio` | `struct<bytes: binary, path: null>` | embedded 24 kHz mono WAV bytes |
| `id`, `text_id` | string | `kss-00000` — `text_id` is the sentence key |
| `voice_id` | string | `KSS`, must appear in `config.speakers` |
| `text` | string | `expanded_script` |
| `duration_seconds` | double | |
| `sample_rate` | int64 | 24000 |
| `frames` | int64 | samples |
| `token_count` | int64 | front-end token length |
| `source_license` | string | `CC BY-NC-SA 4.0` |
| `source_url` | string | HF dataset URL |
| `split` | string | `train` / `validation` / `test` |

**Split policy — group-wise by sentence, seed 42, 95 / 2.5 / 2.5.**
KSS records 391 sentences more than once; a row-wise split puts two takes of one sentence
in different splits and `annotate` fails with
`ValueError: sentence leakage between train and validation`.
`scripts/resplit_kss.py` assigns every recording of a transcript to one split:

```powershell
.\.venv\Scripts\python.exe scripts\resplit_kss.py --data data --seed 42
```

Front-end gate: `phonemize_mixed(text)` runs on 100 % of the rows with **zero `<unk>`**.

## 2. `features2/` — acoustic features

`tiny_voca.prepare` decodes each clip and computes the DSP targets once, so the aligner and
the trainer never touch the WAV bytes again.

```powershell
.\.venv\Scripts\python.exe -m tiny_voca.prepare --inputs data\train-00000-of-00001.parquet `
  --output features2 --split train --config config-ko.json [--resume]
```

Per split the directory holds `<split>.jsonl` (one row per clip), `<split>.preparation.json`
(`"complete": true` + the acoustic fingerprint) and `<split>/<sha256>.pt`:

| `.pt` key | shape | meaning |
| --- | --- | --- |
| `wave` | `[samples]` | 24 kHz float mono |
| `mel` | `[frames, 80]` | log-mel, hop 256 → **93.75 frames/s** |
| `prosody` | `[frames, 3]` | log F0 (`librosa.pyin`), voicing probability, energy |
| `speaker` | int | index into `config.speakers` |

Manifest row keys: `id, record_key, path, source_file, speaker, text, text_hash, frames,
acoustic, provenance`.

**Parallel preparation.** `prepare` is single-process and `pyin`-bound (~100 clips/min), so
12.8 k clips cost ~2 h. Split the shard and run 6 processes (22 min total), then merge:

```powershell
.\.venv\Scripts\python.exe scripts\split_shards.py data\train-00000-of-00001.parquet data_parts\train-{index}.parquet 4
0..3 | ForEach-Object { .\.venv\Scripts\python.exe -m tiny_voca.prepare --inputs "data_parts\train-$_.parquet" --output "features_w\train-$_" --split train --config config-ko.json }
.\.venv\Scripts\python.exe scripts\merge_features.py features2 train features_w\train-0 features_w\train-1 features_w\train-2 features_w\train-3
.\.venv\Scripts\python.exe scripts\merge_features.py features2 validation features_w\validation
.\.venv\Scripts\python.exe scripts\merge_features.py features2 test features_w\test
```

`merge_features.py` re-keys `record_key` to `<part>:<index>` so journal ids stay unique;
feature `.pt` names hash the clip id, so parts never collide. It fails loudly if a part is
missing `.pt` files for rows in its manifest.

Accepted: **12 210 + 322 + 322, rejected 0**.

## 3. `annotations_full/` — Korean forced alignment

`tiny_voca.annotate` aligns the front-end tokens to the mel frames and writes the durations
the trainer consumes.

```powershell
.\.venv\Scripts\python.exe -m tiny_voca.annotate --features features2 --output annotations_full `
  --dataset-revision kss-24k-v1 --splits train validation test --config config-ko.json --device cuda `
  --max-weak-phone-fraction 0.6 --min-token-probability 0.3 --max-phone-frames 90
```

**Aligner.** `kresnik/wav2vec2-large-xlsr-korean` — 1 205 labels (1 202 Hangul syllables +
`|` + `[UNK]`/`[PAD]`, blank 1 204). CTC Viterbi runs at **syllable** granularity on the
93.75 frames/s grid; each syllable span is then sub-allocated to its NFD Jamo
(onset / nucleus / coda) using the energy + voicing contour (`nucleus_drop_db = 6`).
`facebook/mms-1b-all` was rejected: its `kor` vocabulary has 1 330 symbols but the CTC head
only exposes 154 classes.

**Structural invariants** (asserted per clip):
`len(tokens) == len(stress) == len(durations)`, `sum(durations) == frames`,
every spoken Jamo `≥ 1` frame.

| split | accepted | rejected | accepted hours |
| --- | --- | --- | --- |
| train | 8 504 | 3 706 | **8.448** |
| validation | 234 | 88 | 0.234 |
| test | 228 | 94 | 0.230 |

8 min on the GPU (33 clips/s). Train rejection reasons: `overlong_phone` 2 000,
`weak_alignment` 1 766, `low_confidence` 1 258, `vowels_not_voiced` 134.

Manifest row keys: `id, speaker, split, text, text_hash, tokens, stress, symbols,
durations, frames, path, frontend, acoustic, annotation, qc, provenance, unknown`.
`tokens`/`durations` are the aligned `UNIFIED_VOCAB` ids and their frame counts;
`path` points back into `features2/`.

`qc` block (per clip, drives the gates): `token_probability_mean`, `weak_phone_fraction`,
`max_phone_frames`, `one_frame_phone_fraction`, `phones_per_second`, `pause_frames`,
`leading_frames`, `trailing_frames`, `vowel_voiced`, `voiceless_voiced`, `pause_voiced`,
`pause_level_db`. Typical accepted clip: vowel frames voiced 88–92 %, pause tokens at
−56 dB against −9.5 dB speech, 11–15 phones/s.

**QC limits are tuned for a 1 205-way softmax** (probabilities are lower than an
English 40-phone model's). The defaults reject too much Korean; `0.6 / 0.3 / 90` keeps 70 %
of the corpus and removes every `overlong_phone` case that the 20-clip gate found.

## 4. Regenerating from scratch

```powershell
# 1) Parquet (needs the HF dataset; cached under weights/)
.\.venv\Scripts\python.exe scripts\prepare_kss.py
.\.venv\Scripts\python.exe scripts\resplit_kss.py --data data --seed 42

# 2) features (parallel) + merge          — see the commands above
# 3) annotations                          — see the command above
```

Expected wall clock: 20 min + 22 min + 8 min. Disk: Parquet 1.8 GB, `features2/` 5.5 GB
(12 854 `.pt` files), `annotations_full/` 32 MB.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `sentence leakage between train and validation` | re-run `scripts/resplit_kss.py` |
| `resume inputs, feature schema or admission limits changed` | the shard was rewritten; re-prepare, or merge from `features_w/` |
| `annotation directory exists; use --resume or a new directory` | identity changed → new `--output` |
| `acoustic preparation for <split> is missing or incomplete` | `<split>.preparation.json` lacks `"complete": true`; finish that split |
| `<unk>` spikes in the front end | a punctuation/symbol outside `UNIFIED_VOCAB`; extend `korean.py` normalisation |
| prepare is slow | it is `pyin`-bound; shard it across processes as above |
