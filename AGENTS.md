# AGENTS.md — working in the Korean TTS workspace

Operational guide for an agent working in `C:\ai-workspace\korean-tts`.
The pipeline is **built and trained**; a 130 000-step Korean + code-mixed model exists at
`runs/full/inference.pt`. Read this first, then `GUIDE.md` (deliverables & testing),
`DATASET.md` (data), `TRAIN.md` (training), and `FINAL_REPORT.md` / `WORKLOG.md` (results
and the full decision log).

## Mission

Build, adapt, verify, train and evaluate a 24 kHz Korean TTS pipeline on
[`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca) +
[`n4ze3m/dizione`](https://github.com/n4ze3m/dizione) using `Bingsu/KSS_Dataset`
(CC BY-NC-SA 4.0), producing verified, intelligible 24 kHz Korean `.wav` samples.
"English stays English; Korean stays Korean" — English spans go to dizione, Korean spans
to g2pk, in one 142-token vocabulary.

## Operating rules

1. **Inspect before editing** — read the source before assuming paths, flags, shapes or
   schemas. `tiny-voca/src/tiny_voca/*.py` is small and worth reading.
2. **Fail fast, verify every gate** — each stage below has a check; do not chain the next
   stage onto an unchecked one.
3. **Log to `WORKLOG.md`** — commands, timings, gate outputs, decisions and their reasons.
4. **No silent shortcuts** — real acoustic forced alignment (CTC/Wav2Vec2/MMS/MFA);
   label uniform duration splitting as a fallback if you ever use it.
5. **Smoke before full** — 100 clips / 500–1 000 steps before the hours-long run.

## Environment

| | |
| --- | --- |
| Hardware | RTX 4090 24 GB (**~18 GB usable**), Ryzen 9 5900X (24 threads), 39.9 GiB RAM |
| OS / runtime | Windows 11, Python 3.11.16, CUDA 12.6, torch 2.14.1+cu126, transformers 4.57.6, g2pk 0.9.4, faster-whisper, jiwer |
| Layout | The workspace root is a git repo (`main` → `github.com/ooodev/korean-tts`, 75 MB); `tiny-voca/` and `dizione/` are vendored as plain files. Committed: `inference.pt`, `annotations_full/`, `weights/phonemizer/`, `logs_train_full.txt`, `samples/`, `eval/full/`, `eval/precheck/`. Not committed (regenerable): `data/`, `features2/`, `features_en/`, `features_mixed/` (junctions), `runs/full/last.pt` + `step-*.pt` |

Run everything from the workspace root with the venv interpreter:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; $env:PYTHONWARNINGS="ignore"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"
$env:TINY_VOCA_FRONTEND="ko"          # must be set before importing tiny_voca
.\.venv\Scripts\python.exe -m tiny_voca.<stage> …
```

Training also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the GAN phase
fragments the caching allocator otherwise).

### Environment gotchas (all hit in this run)

- **The venv has no `pip`.** Install with `uv`:
  `$env:VIRTUAL_ENV="C:\ai-workspace\korean-tts\.venv"; uv pip install <pkg>`.
- **No `pytest`** — use `.\.venv\Scripts\python.exe -m unittest discover -s tests`
  (run from `tiny-voca/`). Current state: **67 OK, 2 skipped**.
- The global `hf` CLI is broken; use `.\.venv\Scripts\hf.exe`.
- `faster-whisper` needs `torch/lib` on the DLL path for `cublas64_12.dll`;
  `scripts/asr_check.py` already does `os.add_dll_directory`.
- PowerShell 5.1 writes `*>>` as UTF-16LE and `Out-File -Encoding utf8` as UTF-8, so a
  driver log mixes both — `scripts/parse_train_log.py` decodes UTF-8 and drops NULs.
- `array -ge scalar` in PowerShell returns an array (always truthy). Cast scalars
  explicitly; this silently ended a training chunk loop early.
- **Demo assets are cached by `VERSION` in `web/public/worker.js`** (Cache Storage key
  `tiny-voca-ko-<VERSION>`). Change anything under `assets/` and bump it, or browsers keep
  the old model.
- GitHub Pages on the free plan needs the repo **public**; `actions/configure-pages` needs
  `enablement: true` to create the site on the first run.
- `scripts/check_shim.py` reads the dictionary the page ships, so set `MECAB_DICT` to
  `tiny-voca\web\public\assets\mecab-dict.json`; without it the shim runs on its tables
  alone and scores ~90 %.

## Pipeline, in order

| # | Stage | Command | Cost |
| --- | --- | --- | --- |
| 1 | Parquet build | `python scripts/prepare_kss.py` | ~20 min |
| 2 | Sentence-wise split | `python scripts/resplit_kss.py` | 12 s |
| 3 | Acoustic features | `python -m tiny_voca.prepare --inputs … --output … --split … --config config-ko.json` | 22 min on 6 processes |
| 4 | Merge shards | `python scripts/merge_features.py features2 train features_w/train-0 …` | 88 s |
| 5 | Forced alignment | `python -m tiny_voca.annotate --features features2 --output annotations_full …` | 8 min GPU |
| 6 | Train | `powershell -File scripts/train_full.ps1 -Steps 130000` | 11 h 35 m |
| 7 | Evaluate | `python -m tiny_voca.evaluate --checkpoint runs/full/inference.pt …` | 16 s |
| 8 | Samples + gates | `scripts/synthesize_samples.py`, `verify_wavs.py`, `asr_check.py` | ~1 min |
| 9 | Browser demo assets | `scripts/build_mecab_dict.py` then `scripts/build_web_assets.py` (`--export` to re-export ONNX) | ~4 min |

Full flag sets are in `DATASET.md` and `TRAIN.md`. Stage 9 needs
`TINY_VOCA_FRONTEND=ko` and `DIZIONE_DATA` set, like every other stage.

## Source layout you will touch

| File | Role |
| --- | --- |
| `tiny-voca/src/tiny_voca/vocab.py` | `EN_SYMBOLS` (76, exact dizione inventory) → `UNIFIED_VOCAB` (142); `PHONES`, `VOWELS`, `VOICELESS`, `SYMBOL_TO_ID` |
| `tiny-voca/src/tiny_voca/korean.py` | `phonemize_mixed()`, `expand_numbers()`, `EN_SPAN` tokenizer, `units` spans for the aligner |
| `tiny-voca/src/tiny_voca/text.py` | front-end switch (`TINY_VOCA_FRONTEND`), `Phonemized(ids, stress, …, units)` |
| `tiny-voca/src/tiny_voca/kalign.py` | Korean CTC aligner: `KoreanRecognizer`, `assemble_ko`, `align_utterance` |
| `tiny-voca/src/tiny_voca/annotate.py` | picks `kalign` for `ko`, persists/restores `units`, QC gates |
| `tiny-voca/src/tiny_voca/evaluate.py` | Korean-aware ASR (`ko`/`small`), CER next to WER |
| `config-ko.json` | 24 kHz, hop 256, n_fft 1024, 80 mels, F0 50–600, `speakers ["KSS"]`, `vocab_size 142` |
| `tiny-voca/web/public/` | the browser demo: `index.html` (Korean) + `worker.js` (Pyodide + ONNX Runtime Web), `en.html`/`worker-en.js` are the upstream English page |
| `scripts/mecab_shim.py` | pure-Python `python-mecab-ko` stand-in for Pyodide; reads `mecab-dict.json` (`MECAB_DICT` env var) |
| `scripts/*` | workspace helpers (see `GUIDE.md` for the full list) |

## Identity / staleness rules (these cause the confusing errors)

- `prepare --resume` keys its journal on the **input shard's sha256** plus the feature
  schema and admission limits. Rewrite a shard (e.g. `resplit_kss.py`) and every resume
  in that directory is invalid → `ValueError: resume inputs, feature schema or admission
  limits changed`.
- `annotate` hashes the per-split manifests, the front-end fingerprint, the aligner and
  the QC settings into an `identity`. Change any of them and you must use a **new
  `--output` directory** → `FileExistsError` / "these annotations are stale".
- `annotate` reports `"complete": true` only when all three splits are present.
- `annotate` rejects transcripts shared by two splits → the KSS split is **group-wise by
  sentence** (12 854 rows, 12 463 distinct sentences).
- `train --resume` restores weights, optimizer, critics, sampler and RNG. `steps` is *not*
  part of the checkpoint recipe, so resuming with a larger `--steps` is safe.

## Verification commands

```powershell
cd tiny-voca; ..\.venv\Scripts\python.exe -m unittest discover -s tests   # 67 OK, 2 skipped
cd ..
.\.venv\Scripts\python.exe scripts\check_units.py         # aligner unit windows, 8 cases
.\.venv\Scripts\python.exe scripts\verify_wavs.py "samples/*.wav"
.\.venv\Scripts\python.exe scripts\check_guide.py         # guide.html placeholders + audio paths
.\.venv\Scripts\python.exe scripts\parse_train_log.py logs_train_full.txt --every 10000 --validation
.\.venv\Scripts\python.exe scripts\checkpoint_probe.py --run runs/full   # per-checkpoint listening set
.\.venv\Scripts\python.exe scripts\check_onnx.py           # ONNX graph == torch model (demo)
$env:MECAB_DICT="C:\ai-workspace\korean-tts\tiny-voca\web\public\assets\mecab-dict.json"
.\.venv\Scripts\python.exe scripts\check_shim.py --sentences 644   # shim == MeCab, held-out splits
```

## Further training — growing KSS's style repertoire (tips)

The v2 model sits at its data's sweet spot: 4.55 M params, 18.13 h (KSS 8.45 h + F2 5.07 h +
M2 4.61 h), 160 k steps. KO CER is saturated against the ASR floor (human KSS audio scores
5.70 % on the same 228-sentence test split; generated audio 5.35 %). More steps or a bigger
model on the same pool will not move that metric. The remaining gains are **style dimensions
the data never covered**, added as **new KSS data + continued training**.

### Style dimensions (priority order)

| Dimension | Today | How to grow it |
| --- | --- | --- |
| Korean numerals (dates, amounts, temps, populations) | strong — TN + KSS rows; part of the CER error mass on numerals is ASR, not TTS | optional 30–60 min of KSS numeral sentences; judge by ear, not CER |
| English → Korean-style loanwords (every English span) | Mode A + `tiny-voca/src/tiny_voca/data/loanwords.tsv` (462 entries); OOV fallback otherwise | **grow the dictionary first (zero training cost)**; then record code-mixed sentences so co-articulation comes from real audio |
| Special characters / punctuation (read naturally or skip) | g2pk + TN cover the common cases | add punctuation-heavy rows to the new pool; 0-`<unk>` front-end gate before training |
| Emotion / expression | weakest — KSS corpus is one calm style; 2 prosody layers + pitch/energy conditioning exist | only real KSS recordings in varied styles; VC/resynthesis can't fake breathing & micro-dynamics |

### Data strategy (best → worst)

1. **Real KSS recording** — keeps the timbre pure. Calibrate to F2: **~5 h of KSS English** =
   3–4 h English + ~1 h code-mixed + 30–60 min numerals. 24 kHz mono, clean. New recordings
   need speaker consent and keep the CC BY-NC-SA chain.
2. **Voice conversion** (F2/M2 English → KSS timbre) — workable; expect breathing/fine-prosody
   artifacts.
3. **Relabel F2 rows as KSS — rejected**: one speaker embedding cannot hold two timbres;
   Korean output drifts toward a mix (classic multi-speaker failure mode).

### Continued-training recipe (where compute gets wasted if done wrong)

- `--init-from runs/v2/inference.pt --new-data <merged annotations>` — inherits weights + step
  counter + GAN critics; new sampler. `--resume` is only for an identical pool.
- LR keeps halving on its own (`0.5^(step/100000)`): 160 k ≈ ⅓ of base, 200 k = ¼ of base.
  The low LR is what protects the old abilities — do not reheat.
- Old data replays automatically: BucketSampler draws uniformly from the merged pool → with
  +3 k new clips the old 16 451 still receive ≈85 % of every step.
- **Pass-count math** (v2 recipe: 12 000 frames/batch = 128 s of audio): the old pool saw
  ≈315 passes (160 k × 128 s ÷ 18.13 h). In +40 k steps on the merged pool every clip gets
  ≈65–70 more draws, so a new 3 k-clip pool ends at **≈70 cumulative passes vs ≈380 for the
  old abilities — ~1/5 of the exposure**. Fix, in ROI order:
  1. **Up-sample the new clips 3–5× in the manifest** (duplicate rows before merging) →
     ≈150–220 passes in the same 40 k steps, old data still gets ≈50;
  2. more steps (160 k → 250 k ≈ +7 h on the 4090);
  3. LR reheat — last resort, it shakes the old gate.
- Tailwind: the acoustic decoder already encoded English from F2/M2 (≈300 passes); KSS English
  is speaker-adaptation territory — the (KSS, English) mapping + the embedding, not English
  itself — so it converges faster than from scratch.

### Pipeline for a new KSS pool

1. 24 kHz mono wavs → `prepare` into a **new** features dir (`--resume` keys on shard sha256).
2. `annotate` with a **new `--output`** (identity rule) — `kalign` for Korean rows,
   espeak-CTC for English rows, `units` kept for code-mixed rows; per-row QC (0 `<unk>`,
   alignment ratio).
3. Merge with the existing 16 451 rows (apply the up-sampling here) → smoke 100 clips /
   500–1 000 steps → full run `--init-from` + `--new-data`, ~40 k step budget (≈3–4 h).
4. Gate (below) → keep or roll back.

### Keep / roll-back gates

| Gate | Threshold | Why |
| --- | --- | --- |
| KO CER, 228-sentence split | 5.35 % ± 0.3 % | ASR floor ≈ 5.7 % — don't chase CER lower; just don't regress |
| EN WER, KSS-English rows | ≤ 1.5 % | the new ability actually works |
| pause_ratio | 1.16 → toward 1.0 | the measurable "extra pauses" trait; ASR-free |
| duration_ratio / pitch_std | ≈0.975 / ≈5.0 st | no speech-rate or intonation drift |
| Human listening battery | numeral / loanword / punctuation / emotion sentences | timbre drift + prosody — ears decide |

`--init-from` starts a new run → `runs/v2` stays intact for rollback.

### Route the front end before you record, not after

KSS sends English spans through Mode A (loanword G2P: Python → 파이썬). If the new recordings
use **native** English, add a mode routing English spans to dizione — otherwise the model
learns what the front end refuses to render. "파이썬 or paɪθɑːn" is a product decision; the
dictionary, `mecab-dict.json` and web-asset rebuilds (+ `VERSION` bump) follow it.

### Capacity reality check

4.55 M params is intentionally small (browser demo, ~21 MB ONNX). A few hours of data per
speaker scales well for it; a 7 000 h multi-speaker corpus belongs to a larger model — at that
scale the extra hours mostly mean "many voices", and 4.5 M cannot hold them. Don't buy hours
expecting 400× gains.

## Phase status — v1.0 (Korean only, `runs/full`)

| Phase | State | Gate |
| --- | --- | --- |
| 1 Bootstrap & audit | ✅ | hardware + schemas in `WORKLOG.md` |
| 2 Environment | ✅ | dizione + g2pk verified in `.venv` |
| 3 Unified front end | ✅ | `test_ko_en_frontend.py` 24/24, suite 67 OK |
| 4 Dataset | ✅ | 12 210 / 322 / 322 rows, 24 kHz mono, 0 `<unk>` |
| 5 Config + aligner | ✅ | 20-clip gate: 17 accepted, 0 structural failures |
| 6 Smoke | ✅ | 100 clips, 800 steps in 86 s, valid wavs |
| 7 Full prepare/annotate/train | ✅ | 8 504 clips aligned, 130 000 steps, no NaN |
| 8 Samples & quality | ✅ | 7 wavs, mean CER 15.7 % (8.4 % numerals folded) |
| 9 Browser demo | ✅ | ONNX parity (durations exact, corr ≥ 0.982), shim 92.6 % identical / 99.80 % tokens, live at ooodev.github.io/korean-tts |

## Phase status — v2.0 (bilingual, `runs/v2`)

| Phase | State | Gate |
| --- | --- | --- |
| P1 ko-loan front end | ✅ | 185/185 golden cases (`test_frontend_gate.py`), suite 257 OK |
| P2 EN data + features | ✅ | 7 976/702/576 clips, 0 rejected, features_en built |
| P3 EN align + merge | ✅ | annotations_en (espeak-CTC) + annotations_mixed 16 451/933/802, fingerprint 3b930c46d7498d22 |
| P4 Train 160k | ✅ | 12 h 22 m, 4 chunks, 0 failures, 0 NaN; 80k/120k gates PASS (KO 6.01→5.67 %, EN 0.90→0.79 %), D1 no warm start, Q3 = 160k |
| P5 Final gates + batteries | ✅ | final: KO CER **5.35 %** ≤ 12.5 %, EN WER **0.80 %** ≤ 1.5 %, 0 token mismatches; battery text 39/39; listening 16/16 wavs OK |
| P6 Browser demo publish | ✅ | model.onnx 21.44 MB (VERSION 2), check_onnx corr ≥ 0.9826 + strict −61 dB, check_shim 92.64 % identical / 99.80 % tokens; live at ooodev.github.io/korean-tts |
