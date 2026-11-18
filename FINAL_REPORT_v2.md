# Bilingual TTS on `tiny-voca` + `dizione` — final report v2.0

One 4.5M-parameter model that speaks **Korean natively** (voice KSS, Mode A
`ko-loan`) and **English natively** (voices F2/M2, Mode B `own`), with Arabic
numerals pronounced the Korean way and English spans inside Korean sentences
read as Korean loanwords. Hardware: RTX 4090 24 GB, Ryzen 9 5900X (24 threads),
39.9 GiB RAM, Windows 11, Python 3.11.16, CUDA 12.6.

Licences: KSS **CC BY-NC-SA 4.0** — the Korean voice and the model weights
inherit the non-commercial clause. The English voice is trained on the
upstream `n4ze3m/tiny-voca` dataset (F2/M2).

## Summary

| | |
| --- | --- |
| Serving | one checkpoint, two front ends — voice selects the mode: KSS → Mode A (`ko-loan`), F2/M2 → Mode B (`own`) |
| Front end | 142-token unified vocabulary (unchanged from v1); Mode A adds Korean TN + loanword G2P→Jamo before segmentation |
| Corpus | KSS 12 210/322/322 clips (8.45 h) + n4ze3m EN 7 976/702/576 clips; merged manifest **16 451 / 933 / 802** |
| Aligners | `wav2vec2-large-xlsr-korean` CTC (KSS, reusing v1's `annotations_full`) + `wav2vec2-lv-60-espeak-cv-ft` CTC (F2/M2, `own` front end) |
| Model | **4 550 347** parameters, 3 speakers (`KSS`, `F2`, `M2`), **160 000 steps** from scratch (GAN from 8 000) on an RTX 4090 in **12 h 22 m** (4 resumable ≤4 h chunks, 0 failures) |
| Gates | KO CER ≤ 12.5 % (v1: 11.8 %) at the KSS rows, EN WER ≤ 1.5 % with `small.en` at the F2/M2 rows; decision gates at 80k/120k both passed (KO 6.01 % → 5.67 %, EN 0.90 % → 0.79 %); **final PASS: KO CER 5.35 %, EN WER 0.80 %** (0 frontend token mismatches) |
| Batteries | 39-case battery (19 numbers + 20 loanwords, P1 golden tables) + 16-case Mode A+B listening set; text level **39/39 OK**, ASR proxy 58.9 % CER (digit-bias proxy, not the spec gate), listening 16/16 wavs OK |
| Browser demo | voice `<select>` (3 voices), voice=mode routing, 25 example chips in 5 groups (한국어/숫자/한영 혼용/영어 F2/영어 M2, tap-to-play); ONNX + Pyodide in the page — published site measured 2026-11-18: 0 page errors, first-visit load 6.5 s, numbers 4.4 s in 0.28 s (15.7× real time), code-mixed 3.5 s in 0.21 s (16.2×), English (F2) 4.1 s in 0.35 s (11.9×), phonemization 33–149 ms; loanword dictionary confirmed live (Python → 파이썬) |

## 1. Serving modes (one checkpoint, two front ends)

| Mode | Front end | English spans | Numbers | Voice |
| --- | --- | --- | --- | --- |
| **A — "Korean speaker"** (default) | `ko-loan` | loanword G2P → Jamo | Korean TN | KSS |
| **B — "native English"** | `own` | dizione IPA (stress 0/1/2) | English | F2 / M2 |
| C — (reserved, v2.1) | `ko` | dizione IPA under KSS voice | — | KSS |

The two modes are one faithful dispatch (`text.phonemize` under the serving
front end), not two code paths: `ko`/`ko-loan` → `phonemize_mixed`, `own` → the
dizione path. The browser worker, the EN gate, and the annotations all call the
same dispatch — which is what keeps
`frontend_token_mismatches: 0` at every layer.

## 2. Front end (Phase 1)

New mode `ko-loan` and the Korean TN/loanword engine, implemented per
`V2_SPEC.md` §2:

- **Korean TN before segmentation** (`expand_numbers` rework): counters,
  Sino units, clock times, dates, phone numbers, currency, percents, ranges,
  units — the full spec §2.2 table; `0 → 공` in digit-by-digit runs.
- **Loanword G2P** (`loanwords.py`): curated dictionary → acronym letter
  names → OOV fallback (dizione IPA → IPA→Jamo state machine per the
  외래어 표기법 rules) → syllabification → NFD Jamo, stress 0, syllable `units`
  (kalign-compatible). IPA fallback kept < 2 % of prompt tokens.
- `own` and `ko` are byte-unchanged; their tests stayed green throughout.

Verification gate (spec §2.4): **185 golden cases, 100 % pass**
(`tests/test_frontend_gate.py`: 20 sandhi / 75 TN / 52 loanword / 20 Mode A
segmentation / 10+ Mode B regression); suite **257 OK, 2 skipped** (67 in v1,
including 5 new `--serve-frontend` regression tests).

## 3. Data (Phase 2)

| Corpus | Train / Val / Test | Notes |
| --- | --- | --- |
| KSS (KSS voice) | 12 210 / 322 / 322 | reused from v1: sentence-wise split, 24 kHz mono, 0 `<unk>` |
| n4ze3m/tiny-voca (F2/M2) | 7 976 / 702 / 576 | screened per spec §3: Whisper WER ≤ 0.10, 0.5–15 s, < 512 tokens, zero `<unk>` under `own`; 0 rejected at preparation |

Merged manifest (`annotations_mixed`, fingerprint `3b930c46d7498d22`):
**train 16 451 / validation 933 / test 802**. `features_mixed/{ko,en}` are
junctions to `features2` / `features_en`.

## 4. Alignment

- **KSS**: `kalign` (Korean CTC Viterbi at syllable level, energy/voicing
  sub-allocation to Jamo) — v1's `annotations_full` reused verbatim; the
  `units` spans carry the loanword syllables, so Mode A text aligns directly.
- **F2/M2**: `wav2vec2-lv-60-espeak-cv-ft` CTC under the `own` front end —
  the upstream v4 alignment path. Full run 3.3 min: train 7 947/7 976
  (29 rejected), val 699/702, test 574/576, `complete: true`.

## 5. Training (Phase 3)

- `config-v2.json`: KSS/F2/M2, `vocab_size 142`, acoustics identical to
  `config-ko.json` (24 kHz, hop 256, n_fft 1024, 80 mels, F0 50–600).
- **From scratch** (decision D1); the v1 checkpoint stays untouched as the
  baseline. Warm-start fallback staged in advance: `eval/v2/warm-snapshot.pt`
  (v1 weights under the v2 config, step 130 000, fresh critics) for the case
  KO CER > 12.5 % at the 120k gate.
- Recipe unchanged from v1: batch 12 000 frames/48 utterances, warmup 1 000,
  half-life 100 000, GAN 8 000/3 000/0.7, teacher 3 000/7 000/0, 4-hour
  resumable chunks.
- Post-GAN throughput **≈ 3.5 steps/s** (v1: 3.2), step-verified: 160 000 steps
  in 3 × 4 h + 20 min of chunk time (attempts 1–4, no failures).

## 6. Evaluation

Decision gates at 80k/120k and the final gate on `runs/v2/inference.pt` run
`scripts/run_v2_gates.ps1`:

- **KO gate**: KSS rows of the merged test split
  (`--per-voice 300 --novel 0`, 228 rows → all), metric
  `summary.test.KSS.generated_cer`; pass ≤ 12.5 % (v1: 11.8 %).
- **EN gate**: F2/M2 rows of the merged test slice
  (`--per-voice 600`, 574 rows → all), served with
  `TINY_VOCA_FRONTEND=ko-loan` + `evaluate --serve-frontend own` (the
  checkpoint loads only under the training front end; the switch happens after
  data loading, before synthesis), ASR **small.en** per spec §5, metric
  `summary.test.all.generated_wer`; pass ≤ 1.5 %.

**Decision-gate history** (`eval/v2/gate-80000/`, `eval/v2/gate-120000/`):

| Step | KO CER | pass (≤ 12.5 %) | EN WER | pass (≤ 1.5 %) | frontend_token_mismatches |
| --- | --- | --- | --- | --- | --- |
| 80 000 | 6.01 % | ✅ | 0.90 % | ✅ | 0 |
| 120 000 | 5.67 % | ✅ | 0.79 % | ✅ | 0 |

**Q3 (160k vs 190k) outcome: 160 000.** At 120k both metrics are already far
under the bars (KO 5.67 % vs v1's 11.8 %; EN 0.79 % vs the 1.5 % target), still
improving 80k→120k (KO −0.34 pt, EN −0.11 pt) — nothing justifies the reserved
30k extension, so the original budget stands. The 120k pass also closes the D1
question: no warm start was needed, training continued from step 0.

**Final gate** on `runs/v2/inference.pt` (step 160 000, `eval/v2/final/`):

| Gate | Samples | Primary metric | Pass bar | Result |
| --- | --- | --- | --- | --- |
| KO (KSS rows, Mode A) | 228 | generated CER | ≤ 12.5 % | **5.35 %** ✅ (v1: 11.8 %) |
| EN (F2/M2 rows, Mode B) | 574 | generated WER | ≤ 1.5 % | **0.80 %** ✅ |

Per-voice and full metric set:

| Gate | WER | CER | dur ratio | pause ratio | mel DTW | pitch std (gen / ref, st) |
| --- | --- | --- | --- | --- | --- | --- |
| KO · KSS | 18.43 % | **5.35 %** | 0.975 | 1.160 | 0.773 | 4.96 / 5.02 |
| EN · F2 (284) | **0.67 %** | 0.21 % | 0.974 | 1.095 | 0.801 | 3.40 / 3.82 |
| EN · M2 (290) | **0.92 %** | 0.27 % | 0.976 | 1.126 | 0.668 | 4.33 / 4.27 |

`frontend_token_mismatches: 0` on both gates. Two honest observations:
pause ratio ~1.1–1.2 (the model slightly lengthens pauses vs the references)
and F2's pitch spread runs 0.89× the reference (flatter intonation on the
female voice; M2 matches at 1.01×). Both are outside the pass bars, noted for
the listening pass.

## 7. Samples & batteries

- `scripts/battery_v2.py`: the spec §5 listening batteries — 19 number cases +
  20 loanword cases imported from the P1 golden tables, text-level pre-check
  against the locked contract, Mode A KSS synthesis, Whisper-ko CER proxy.
- `scripts/listening_v2.py`: 16-case Mode A+B listening set on the final
  checkpoint (greetings, numbers, code-mix, flow, questions, stress).

**Battery results** (`samples_v2_battery/`, `battery_summary.json`):

- Text level: **39/39** locked readings reproduce exactly (P1 contract holds
  on the final checkpoint).
- ASR proxy (Whisper small, `language=ko`): mean CER **58.9 %** — a noisy
  proxy, not the spec gate (that is the human listening pass). Two dominant
  ASR-side biases inflate it: digit-preference (`세명` spoken, `3명` transcribed;
  `3시 30분` → `3시 30분`, `10km` → `10km`) and short-word instability
  (`AI` → `예, 예, 예`, `bus` → `다시`, `bag` → `big`). 9/20 loanwords were
  transcribed exactly (hello, computer, dollar, coffee, algorithm, chocolate,
  digital, nameplate, xylophone); number cases where ASR cooperated confirm
  the Korean readings (`3번째` → `세번째`, `20살` → `스무살`).
- `scripts/battery_v2.py` fixes this window: `units_of` now reads the
  `(label, start, end)` triples of `Phonemized.units`, and case names are
  deterministic + filesystem-safe (the old name scheme crashed on `3/14`).

**Listening set** (`samples_v2/`, 16 files, `verify_wavs` 16/16 OK): 8 Mode A
KSS (`a1_greeting`…`a8_numbers_units`) + 8 Mode B F2/M2
(`b1_female_greeting`…`b8_male_numbers`); all at 24 kHz mono, peaks 0.44–0.98.

## 8. Browser demo (P5)

- `worker.js`: speak messages carry `voice`/`mode`; voice → mode per §1
  (explicit `mode` overrides). Phonemisation is the faithful dispatch
  (`set_frontend` + `phonemize`) exactly as in the annotations —
  `phonemize_mixed(text, "own")` was deliberately not used: it differs from
  the standard path on quote+comma punctuation (4 of 10 EN corpus sentences).
- `index.html`: voice `<select>` from `config.voices`, 25 example chips in
  5 groups (한국어/숫자/한영 혼용/영어 F2/영어 M2, five each) with tap-to-play
  (6 KSS Mode A + F2/M2 Mode B), bilingual copy, v2 facts.
- Assets: `model.onnx` (21.4 MB) re-exported under `TINY_VOCA_FRONTEND=ko-loan`
  from `runs/v2/inference.pt`; first-visit download 28.8 MB (excluding Pyodide);
  `VERSION` 1 → 2 for Cache Storage.

**Demo gates** (this publish run, `logs_v2_publish.txt`):

- `check_onnx.py`: durations exact, min ONNX↔torch correlation **0.9826**
  (internal noise realisation differs, peak 0.199), strict `--external-noise`
  graph peak **0.000906** (~−61 dB, float32 conv accumulation) → OK.
- `check_shim.py`: held-out val+test, 462 sentences / 17 281 tokens,
  **92.64 %** identical token streams (gate ≥ 90 %), token accuracy
  **99.80 %** (34 tokens differ; the `python-mecab-ko`-only assimilation rules).

**Published site** (https://ooodev.github.io/korean-tts/, headless-Chrome
measurement, 2026-11-18, VERSION 4): 0 page errors, first-visit load 6.5 s, then
from Cache Storage. Per chip (status `완료.`, audio element filled):

| chip | voice / mode | audio | synth wall | real time | phonemization |
| --- | --- | --- | --- | --- | --- |
| 숫자 | KSS / A | 4.4 s | 0.28 s | 15.7× | 33 ms |
| 한영 혼용 | KSS / A | 3.5 s | 0.21 s | 16.2× | 64 ms |
| 영어 | F2 / B | 4.1 s | 0.35 s | 11.9× | 149 ms |

`Python` renders `파ᅵ썬ᅳ로` — the trained dictionary spelling (파이썬), not the
OOV fallback. Two publish-blocking bugs were found and fixed during
measurement (VERSION 3/4): `kofront/loanwords.py` and
`kofront/data/loanwords.tsv` were missing from `pyfront.zip`, so code-mixed
input crashed the worker or fell back off the training distribution; see
WORKLOG 2026-11-18.

## 9. Reproducing

```powershell
# environment
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko-loan"
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"

# 1. data: KSS (v1 shards, sentence-wise split) + screened n4ze3m EN corpus
python scripts/prepare_kss.py            # v1 output reused
python scripts/prepare_en.py             # EN screen -> data_en shards
python -m tiny_voca.prepare ...          # acoustic features (ko + en, separate runs)

# 2. alignment
python -m tiny_voca.annotate ...         # ko (kalign)  -> annotations_full (reused)
python -m tiny_voca.annotate ...         # own (espeak-CTC) -> annotations_en
python scripts/merge_annotations.py      # annotations_mixed (16 451/933/802)

# 3. train (chunked, auto-resume)
powershell -File scripts/train_full.ps1 -Steps 160000 -Config config-v2.json ...

# 4. gates at 80k/120k and final
python scripts/gate_manifest.py ...      # KSS slice / F2+M2 slice
powershell -File scripts/run_v2_gates.ps1 -Checkpoint runs/v2/inference.pt ...

# 5. batteries + samples
python scripts/battery_v2.py ...
python scripts/listening_v2.py ...

# 6. browser demo assets, then serve tiny-voca\web\public (or push: Pages publishes it)
$env:TINY_VOCA_FRONTEND="ko-loan"
python scripts/build_web_assets.py --checkpoint runs/v2/inference.pt --export
python scripts/check_onnx.py; $env:MECAB_DICT="...mecab-dict.json"; python scripts/check_shim.py
```

Full flag sets: `DATASET.md`, `TRAIN.md`, `V2_SPEC.md`; the complete decision
log: `WORKLOG.md`.
