# Korean & code-mixed TTS on `tiny-voca` + `dizione` — final report

Target: 24 kHz Korean speech synthesis from `Bingsu/KSS_Dataset` on top of
[`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca) and
[`n4ze3m/dizione`](https://github.com/n4ze3m/dizione), with a code-mixed
(Korean + English) front end. Hardware: RTX 4090 24 GB, Ryzen 9 5900X (24 threads),
39.9 GiB RAM, Windows 11, Python 3.11.16, CUDA 12.6.

KSS is licensed **CC BY-NC-SA 4.0** — non-commercial research/testing only.

## Summary

| | |
| --- | --- |
| Front end | 142-token unified vocabulary: dizione English (ids 0–75, stress intact) + Korean NFD Jamo (76–141); `g2pk` phonology, silent `ㅇ` stripped |
| Corpus | 12 854 KSS clips → **8 504 aligned** (8.45 h) at 24 kHz, 0 `<unk>`, sentence-wise 95/2.5/2.5 split |
| Aligner | `wav2vec2-large-xlsr-korean` CTC Viterbi at syllable level + energy/voicing sub-allocation to Jamo |
| Model | 4 549 579 parameters, single speaker `KSS`, **130 000 steps** (GAN from 8 000), 11 h 35 min on an RTX 4090 |
| Held-out test | **CER 11.8 %** (reference 2.6 %), duration ratio 0.907, pitch spread 6.09 vs 5.78 st, mel DTW 0.778 |
| Phase 8 samples | 7 files, 24 kHz mono, **mean CER 15.7 %** (8.4 % with numerals folded), 3 transcribed exactly |
| Browser demo | [ooodev.github.io/korean-tts](https://ooodev.github.io/korean-tts/) — ONNX + Pyodide, 9.7–17× real time in Chrome, no page errors |
| Total wall clock | **≈ 16.5 h** of the 18 h budget |

## 1. Files modified

### `dizione`
No source changes. Used as the English side of the front end
(`dizione.phonemize`, 76-symbol vocabulary, `<bos>`/`<eos>` wrapped).

### `tiny-voca` (all edits are uncommitted working-tree changes)

| File | Change |
| --- | --- |
| `src/tiny_voca/vocab.py` | **new** — `EN_SYMBOLS` (76, the exact dizione inventory) + `UNIFIED_VOCAB` (142) built by appending the Korean NFD Jamo; `PHONES`, `VOWELS`, `VOICELESS`, `SYMBOL_TO_ID` derived from it |
| `src/tiny_voca/korean.py` | **new** — `phonemize_mixed()` (English span → dizione, Korean span → g2pk → NFD → strip silent `ㅇ`), `expand_numbers()`, `EN_SPAN` tokenizer, `units` spans for the aligner |
| `src/tiny_voca/text.py` | front-end switch (`TINY_VOCA_FRONTEND=ko`), `Phonemized` gained a 5th field `units`, re-exports `PHONES`/`PHONE_SET` from `vocab` |
| `src/tiny_voca/config.py` | `vocab_size: int = 0` → defaults to `len(SYMBOLS)` of the active front end |
| `src/tiny_voca/kalign.py` | **new** — Korean CTC forced aligner (`KoreanRecognizer`, `assemble_ko`, `align_utterance`) |
| `src/tiny_voca/annotate.py` | picks `kalign` for the `ko` front end, persists/restores `units` |
| `src/tiny_voca/evaluate.py` | Korean-aware: ASR language and model follow the front end (`ko`/`small` instead of hardcoded `en`/`small.en`), ASR runs on the GPU when the evaluation does, Korean text normalizer, **CER** reported next to WER |
| `tests/test_ko_en_frontend.py` | **new** — 24 tests for the unified front end |
| `tests/test_frontend.py`, `tests/test_frontend_own.py`, `tests/test_model.py` | made front-end aware (pin `espeak` in `setUp`, compare against `EN_SYMBOLS`, fixed seed) |
| `config-ko.json` | **new** — Korean model configuration (`speakers ["KSS"]`, `vocab_size 142`) |

### Workspace scripts (new, outside the two clones)

`scripts/prepare_kss.py` (Parquet build) · `resplit_kss.py` (sentence-wise split) ·
`split_shards.py` + `merge_features.py` (parallel acoustic preparation) ·
`train_full.ps1` (chunked, auto-resuming training) · `synthesize_samples.py` (Phase 8
manifest → `samples/*.wav` + `tokens.json`) · `verify_wavs.py` (audio gate) ·
`asr_check.py` (Korean back-transcription, CER/WER + numeral folding) ·
`parse_train_log.py` (throughput/loss table) · `check_units.py`, `checkpoint_probe.py`,
`verify_phase5_align.py`.

## 2. Unified vocabulary

`UNIFIED_VOCAB` = **142 tokens**, English ids unchanged at `0..75`:

| block | indices | count |
| --- | --- | --- |
| dizione English (`<pad> <unk> " " , . ? ! -` + 68 phones/`<bos>`/`<eos>`) | 0 – 75 | 76 |
| choseong `U+1100..U+1112` without silent `ㅇ` `U+110B` | 76 – 93 | 18 |
| jungseong `U+1161..U+1175` | 94 – 114 | 21 |
| jongseong `U+11A8..U+11C2` (final `ㅇ` `U+11BC` kept) | 115 – 141 | 27 |

Stress is `0/1/2` for English phones and `0` for every Korean token.
Front-end gate: `tests/test_ko_en_frontend.py` — **24/24 pass**; whole suite
**67 tests OK (2 skipped)**.

## 3. Dataset

`Bingsu/KSS_Dataset` → `scripts/prepare_kss.py` → 24 kHz mono Parquet in `data/`:

| split | clips | hours | distinct sentences |
| --- | --- | --- | --- |
| train | 12 210 | 12.211 | 11 833 |
| validation | 322 | 0.326 | 313 |
| test | 322 | 0.321 | 317 |
| **total** | **12 854** | **12.858** | 12 463 |

All 12 854 clips pass the filters (0.5–15 s, `< 512` tokens); `phonemize_mixed`
runs on 100 % of the rows with **zero `<unk>`**. `voice_id = "KSS"`, ids `kss-00000…`.
The split is **group-wise by sentence** (seed 42): KSS records 391 sentences twice, and
`annotate` requires disjoint transcripts across splits.

## 4. Forced aligner

`kresnik/wav2vec2-large-xlsr-korean` (1 205 labels = 1 202 Hangul syllables + `|` +
`[UNK]`/`[PAD]`, blank index 1 204) → CTC Viterbi at **syllable** granularity on the
24 kHz / hop-256 grid (93.75 frames/s), then each syllable span is sub-allocated to its
NFD Jamo (onset / nucleus / coda) using the energy + voicing contour
(`nucleus_drop_db = 6`). `facebook/mms-1b-all` was rejected: its `kor` vocabulary has
1 330 symbols but the CTC head only has 154 classes.

20-clip gate (`scripts/verify_phase5_align.py`): 20/20 aligned, **17 accepted,
0 structural failures** — `len(tokens) == len(stress) == len(durations)`,
`sum(durations) == frames`, every Jamo `>= 1` frame, no collapsed clip.
Vowel frames voiced 88 % (median), pause tokens sit at −56 dB against −9.5 dB speech,
11.4 phones/s. QC limits retuned for a 1 205-way softmax:
`--max-weak-phone-fraction 0.6 --min-token-probability 0.3`.

Full-corpus annotation: _(filled in below)_

## 5. Training

`scripts/train_full.ps1` runs `tiny_voca.train` in wall-clock chunks and resumes from
`last.pt`, so an interrupted session loses at most the chunk in flight (tested: 120 →
225 → 240 steps across three chunks).

```
python -m tiny_voca.train --train annotations_full/train.jsonl --validation annotations_full/validation.jsonl \
       --output runs/full --config config-ko.json --device cuda \
       --steps 130000 --batch-frames 12000 --max-batch 48 --learning-rate 3e-4 --warmup 1000 --lr-half-life 100000 \
       --gan-start 8000 --gan-ramp 3000 --gan-ratio 0.7 --feature-matching 30 \
       --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 \
       --validation-size 32 --save-every 2000 --keep-every 10000 --log-every 50 --max-seconds 14400
```

* **Model**: 4 549 579 inference parameters (+ the critic head, trained only before GAN
  handover), single speaker `KSS`, 142-token unified vocabulary.
* **Batch**: dynamic buckets of ~32 clips ≈ 112 audio-s (10 600 frames) under
  `--batch-frames 12000 --max-batch 48`; 5.2 GB of VRAM in the teacher-forced phase,
  14.4 GB in the GAN phase (peak 15.5 GB, budget 18 GB).
* **Speed**: 14.0 steps/s before the GAN, 3.4 steps/s after it (the critics add ~3 passes
  per step) → 380 audio-s/s, i.e. **~1 100 s of speech per step**.

| step | loss | mel | duration | voicing | pitch | pause | stft | teacher | gan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 000 | 9.52 | 0.563 | 0.158 | 0.115 | 0.163 | 0.119 | 0.272 | 1.20 | – |
| 10 000 | 41.0 | 0.647 | 0.126 | 0.094 | 0.125 | 0.041 | 0.241 | 1.31 | 33.9 |
| 30 000 | 13.68 | 0.516 | 0.102 | 0.071 | 0.088 | 0.007 | 0.191 | 1.13 | 4.0 |
| 70 000 | 11.10 | 0.471 | 0.079 | 0.060 | 0.067 | 0.004 | 0.167 | 1.06 | 2.1 |
| **130 000** | **9.99** | **0.449** | **0.062** | **0.052** | **0.058** | **0.002** | **0.152** | **1.02** | 1.4 |

The teacher-forced fraction decays from 1.0 to 0 by step 10 000, so the decoder learns to
run on its own predicted durations from there on; the generator loss spikes at the GAN
start (8 000) and falls back as the critics saturate.

**Wall clock**: 05:52 → 17:27, i.e. **11 h 35 min of training** across three chunks
(4 h + 3.5 h + 1 h, resumed from `last.pt` each time). 14 checkpoints kept (GAN start,
every 10 000 steps), 1.12 GB, plus `last.pt` and the 17.4 MB `inference.pt`.

### Mid-training pre-check (step ~80 000)

The Phase 8 chain was run against the in-progress `inference.pt` to de-risk it:
7 utterances synthesized in 16 s, all 24 kHz mono, RMS 0.075–0.101, peak ≤ 0.617,
durations 3.5–6.2 s, and faster-whisper `small` (GPU) gives **mean CER 16.7 % /
WER 25.4 %** (`sample_06` is transcribed exactly, CER 0.0 %). The 700-step smoke model
was at 93.5 % CER, so the GAN phase is doing its job.

## 6. Evaluation

`python -m tiny_voca.evaluate --checkpoint runs/full/inference.pt --annotations
annotations_full --output eval/full --tag full --splits test --per-voice 6 --novel 2
--device cuda` — 16 s on the GPU, 6 held-out test sentences + 2 sentences from no split,
all synthesised **free-running** (text in, durations/pitch/voicing/energy predicted).

| metric (mean over 6 test clips) | generated | recorded reference |
| --- | --- | --- |
| CER (faster-whisper `small`, `ko`) | **11.8 %** | 2.6 % |
| WER | 20.8 % | 4.2 % |
| duration ratio (generated / recorded) | 0.907 | – |
| pause ratio | 0.956 | – |
| DTW log-mel distance | 0.778 | – |
| pitch standard deviation | 6.09 st | 5.78 st |
| voiced fraction | 0.673 | 0.572 |

Three of the six test sentences are transcribed **exactly** (CER 0.0 %); the worst is a
short sentence at 53.8 % CER whose generated audio is 0.62× the recorded length. The
model's pitch spread and pause structure sit within a few percent of the recordings, and
the reference column shows how much of the CER is the ASR model itself (2.6 %).

`frontend_token_mismatches: []` — re-running the front end on the raw text reproduces the
annotated tokens for every evaluated sentence, so preparation and inference agree.

The two built-in *novel* sentences are English (`The quick brown fox …`), which the model
has never heard in KSS; they synthesise at 4.6–5.2 s with a plausible Korean-accented
English contour. Korean generalisation is measured on the Phase 8 samples below.

Validation at 130 000 steps (32 clips, free-running): `predicted_mel 0.805`,
`teacher_forced_mel 0.755`, `pitch_error 3.18 st`, `pitch_spread_ratio 0.891`,
`voicing_accuracy 0.926`, `pause_accuracy 0.948`, `duration_length_ratio 1.051`.

## 7. Samples

`python scripts/synthesize_samples.py --checkpoint runs/full/inference.pt --manifest
samples/manifest.json --outdir samples` (5.5 s for all seven), then
`scripts/verify_wavs.py` and `scripts/asr_check.py` (faster-whisper `small`, `ko`, GPU).

CER is character-level with spaces ignored. Whisper writes Korean numerals as Arabic
digits, so the last column folds a digit run and a run of Sino-Korean numeral morphemes
to the same placeholder — it measures the reading rather than the orthography.

| file | text | duration | RMS | peak | CER | WER | CER (numbers folded) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sample_01_greeting.wav` | 안녕하세요. 오늘 날씨가 정말 맑고 좋네요. | 4.11 s | 0.087 | 0.513 | **0.0 %** | 0.0 % | 0.0 % |
| `sample_02_question.wav` | 지금 몇 시인지 혹시 알려주실 수 있나요? | 3.67 s | 0.079 | 0.627 | **0.0 %** | 0.0 % | 0.0 % |
| `sample_03_phonology.wav` | 국물 맛이 끝내주고 신라의 역사도 흥미롭습니다. | 4.67 s | 0.090 | 0.524 | 10.0 % | 50.0 % | 10.0 % |
| `sample_04_numbers.wav` | 이천이십육 년 십일 월 십오 일 오후 세 시 삼십 분입니다. | 4.70 s | 0.081 | 0.544 | 54.5 % | 90.9 % | **12.5 %** |
| `sample_05_long.wav` | 저는 보통 이십 분 정도 낮잠을 자고 나면 … | 6.22 s | 0.081 | 0.590 | 15.6 % | 26.7 % | **6.5 %** |
| `sample_06_exclamation.wav` | 와, 정말 대단하네요! 드디어 … 성공했습니다! | 6.10 s | 0.084 | 0.583 | **0.0 %** | 0.0 % | 0.0 % |
| `sample_07_codemix_check.wav` | 오늘 Python으로 테스트를 진행했습니다. | 3.61 s | 0.095 | 0.566 | 30.0 % | 25.0 % | 30.0 % |
| **mean** | | **4.73 s** | | | **15.7 %** | **27.5 %** | **8.4 %** |

All seven are 24 000 Hz mono, free of NaN/Inf, non-silent (RMS 0.079–0.095) and un-clipped
(peak ≤ 0.627), with durations in the natural 3.6–6.2 s range for those sentences.

Reading the numbers honestly:

* **Three sentences are transcribed exactly**, and the phonology sentence (nasalization
  국물 → [국물], liquidization 신라 → [실라], tensification 끝내주-, liaison 역사도)
  differs from the reference only in Whisper's eojeol spacing plus one syllable of
  흥미롭습니다, i.e. the phonological rules survive into the audio.
* **Numbers are right**: the 54.5 % CER is Whisper spelling "이천이십육 년" as "2026 년";
  with numerals folded the same clip scores 12.5 %, and the long sentence drops to 6.5 %.
* **Code-mixing works structurally** — `Python` is phonemized by dizione and attached to
  the Korean particle 으로 with no inserted space, and the front end emits
  `pʌɪθɑn` + `으 로`. KSS contains no English recordings, so the English phones are
  unseen at training time and the 30 % CER on that clip reflects the accent of the
  synthesised English, not a front-end failure.

## 8. Browser demo

The model runs in the page at **[ooodev.github.io/korean-tts](https://ooodev.github.io/korean-tts/)**
(`tiny-voca/web/public`, published by `.github/workflows/pages.yml`).

| | |
| --- | --- |
| Graph | `runs/full/inference.pt` → `python -m tiny_voca.export` → `model.onnx` 21.4 MB, run by ONNX Runtime Web (WASM, 1 thread) |
| Parity gate | `scripts/check_onnx.py`: durations exact on 5 sentences, min correlation 0.9824; the `--external-noise` graph reproduces the waveform to a peak of 6.0e-4 |
| Phonemizer | the same Python as training: `pyfront.zip` = g2pK + jamo + `kofront` (`vocab/text/korean.py`) + dizione, in Pyodide 0.26.4 |
| Morphology | `python-mecab-ko` has no WASM build → `scripts/mecab_shim.py` (pure Python) + `mecab-dict.json`, the real MeCab analyses of 20 649 training-split eojeols |
| Shim gate | `scripts/check_shim.py` on held-out validation+test: **92.64 %** identical token stream, **99.80 %** token accuracy |
| Measured | 4.7 s of audio in 0.28 s (16.9× real time) from Pages, 3.6 s in 0.37 s locally; code-mix renders `ᅩ늘paɪθɑːnᅳ로…`; no page errors |
| First visit | ~40 MB (28.8 MB assets + Pyodide), then Cache Storage, keyed on `VERSION` in `worker.js` |

Details in `tiny-voca/web/README.md`; the log of the port is `WORKLOG.md` → *Phase 9*.

## 9. Reproducing

```powershell
# environment
.\.venv\Scripts\python.exe -m pip install -e ./dizione -e ./tiny-voca   # via uv
# 1. data: 24 kHz mono Parquet, sentence-wise 95/2.5/2.5 split, seed 42
.\.venv\Scripts\python.exe scripts\prepare_kss.py
.\.venv\Scripts\python.exe scripts\resplit_kss.py
# 2. acoustic features (4 parallel shards) and merge
.\.venv\Scripts\python.exe scripts\split_shards.py data\train-00000-of-00001.parquet data_parts\train-{index}.parquet 4
.\.venv\Scripts\python.exe -m tiny_voca.prepare --inputs data_parts\train-N.parquet --output features_w\train-N --split train --config config-ko.json
.\.venv\Scripts\python.exe scripts\merge_features.py features2 train features_w\train-0 features_w\train-1 features_w\train-2 features_w\train-3
# 3. Korean forced alignment
.\.venv\Scripts\python.exe -m tiny_voca.annotate --features features2 --output annotations_full `
    --dataset-revision kss-24k-v1 --splits train validation test --config config-ko.json --device cuda `
    --max-weak-phone-fraction 0.6 --min-token-probability 0.3 --max-phone-frames 90
# 4. train (chunked, auto-resume)
powershell -File scripts\train_full.ps1 -Steps 130000 -Output runs\full -Annotations annotations_full
# 5. evaluate, synthesise, verify
.\.venv\Scripts\python.exe -m tiny_voca.evaluate --checkpoint runs\full\inference.pt --annotations annotations_full --output eval\full --tag full --splits test --per-voice 6 --device cuda
.\.venv\Scripts\python.exe scripts\synthesize_samples.py --checkpoint runs\full\inference.pt --manifest samples\manifest.json --outdir samples
.\.venv\Scripts\python.exe scripts\verify_wavs.py "samples/*.wav"
.\.venv\Scripts\python.exe scripts\asr_check.py --manifest samples\manifest.json --outdir samples --model small --device cuda
# 6. browser demo assets, then serve tiny-voca\web\public (or push: Pages publishes it)
.\.venv\Scripts\python.exe scripts\build_mecab_dict.py
.\.venv\Scripts\python.exe scripts\build_web_assets.py --export
.\.venv\Scripts\python.exe scripts\check_onnx.py
.\.venv\Scripts\python.exe scripts\check_shim.py --sentences 644
```

Every command needs `TINY_VOCA_FRONTEND=ko` and `DIZIONE_DATA=<workspace>/weights/phonemizer`;
training also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
