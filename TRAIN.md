# TRAIN.md — model, recipe, throughput, monitoring

Everything about the training run that produced `runs/full/inference.pt`
(130 000 steps, 11 h 35 m on an RTX 4090). Data side is in `DATASET.md`.

## Model

`config-ko.json` is the single source of truth; it is copied into every run directory.

```json
{ "sample_rate": 24000, "hop": 256, "n_fft": 1024, "n_mels": 80,
  "f0_min": 50.0, "f0_max": 600.0, "speakers": ["KSS"], "vocab_size": 142 }
```

* **Grid** — hop 256 at 24 kHz = 10.67 ms/frame = **93.75 frames/s**; 80 log-mel bins.
* **Vocabulary** — 142 tokens: dizione English 76 (ids 0–75, stress 0/1/2) + Korean NFD
  Jamo 66 (choseong 18 excl. silent `ㅇ`, jungseong 21, jongseong 27, stress 0).
* **Size** — 4 549 579 inference parameters; the critic head is training-only.
* **Speaker** — single voice `KSS` (index 0).
* **Targets** — mel + duration + voicing + pitch + pause + energy; pitch/voicing/energy are
  predicted, so inference needs text only.

## Recipe

```powershell
.\.venv\Scripts\python.exe -m tiny_voca.train `
  --train annotations_full/train.jsonl --validation annotations_full/validation.jsonl `
  --output runs/full --config config-ko.json --device cuda `
  --steps 130000 --batch-frames 12000 --max-batch 48 `
  --learning-rate 3e-4 --warmup 1000 --lr-half-life 100000 `
  --gan-start 8000 --gan-ramp 3000 --gan-ratio 0.7 --feature-matching 30 `
  --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 `
  --validation-size 32 --save-every 2000 --keep-every 10000 --log-every 50 `
  --max-seconds 14400
```

| flag | value | why |
| --- | --- | --- |
| `--batch-frames` / `--max-batch` | 12 000 / 48 | VRAM probe (below): flat throughput vs batch size, so a bigger batch buys 3× the audio per step at the same speed |
| `--gan-start` / `--gan-ramp` / `--gan-ratio` | 8 000 / 3 000 / 0.7 | mel/duration supervision first, then adversarial training on 70 % of the batch |
| `--teacher-hold` / `--teacher-ramp` / `--teacher-final` | 3 000 / 7 000 / 0 | teacher forcing held to 3 000, decayed to 0 by 10 000 → the decoder runs on its own durations afterwards |
| `--lr-half-life` | 100 000 | decay matched to the step budget |
| `--validation-size` | 32 | validation runs on **every** save (65 times over the run, ~13 s each) |
| `--save-every` / `--keep-every` | 2 000 / 10 000 | `last.pt` + `inference.pt` every 2 000; long-lived `step-*.pt` every 10 000 |
| `--max-seconds` | 14 400 | wall-clock chunk so a dropped session loses one chunk |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required in the GAN phase.

## Throughput & VRAM (44-clip probe, 300 steps, GAN from 100)

| `--batch-frames` / `--max-batch` | peak VRAM | steady steps/s | audio-s/s | note |
| --- | --- | --- | --- | --- |
| 4 000 / 8 (smoke) | ~9 GB | 8.4 | 358 | pre-GAN |
| **12 000 / 48** | **15 458 MiB** | 2.9–3.2 | **372** | chosen |
| 14 000 / 56 | 17 335 MiB | 1.95 | 291 | more VRAM, slower |
| 16 000 / 64 | 18 589 MiB | – | – | OOM inside the GAN phase |

In the real run, with the CPU free, the phases separate cleanly:

| phase | steps/s | audio-s/s | VRAM |
| --- | --- | --- | --- |
| teacher-forced (1 – 8 000) | 14.0 | ~1 560 | 5.2 GB |
| GAN (from 8 000, full at 11 000) | 3.42 | ~375 | 14.4 GB (peak 15.5) |

A step consumes ~112 audio-s (≈ 32 clips, 10 600 frames). Budget arithmetic:
8.45 h of aligned audio → 1 epoch ≈ 81 s at 372 audio-s/s, so 130 000 steps ≈ 760 epochs.

## Running it in chunks

`scripts/train_full.ps1` loops `tiny_voca.train --resume runs/full/last.pt` until
`--Steps` is reached, and retries a chunk that dies:

```powershell
powershell -File scripts\train_full.ps1 -Steps 130000 -ChunkSeconds 14400 `
  -Output runs/full -Annotations annotations_full -Config config-ko.json -Log logs_train_full.txt
```

This run took three chunks (4 h + 3.5 h + 1 h): 54 996 → 95 063 → 122 000 → 130 000.
`--steps` is not part of the checkpoint recipe, so extending is a one-liner:

```powershell
powershell -File scripts\train_full.ps1 -Steps 200000 -Output runs/full -Annotations annotations_full
```

## Monitoring

```powershell
.\.venv\Scripts\python.exe scripts\parse_train_log.py logs_train_full.txt --every 10000 --validation
.\.venv\Scripts\python.exe scripts\parse_train_log.py logs_train_full.txt --tail 1     # last record + validation block
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

`runs/full/metrics.jsonl` carries the same records. Note that `frames` and `utterances` are
**summed over the log window** (every 50 steps), not cumulative — the instantaneous rate is
`frames / window-seconds`, which is what `parse_train_log.py` prints.

### Loss trajectory

| step | loss | mel | duration | voicing | pitch | pause | stft | teacher | gan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 000 | 9.52 | 0.563 | 0.158 | 0.115 | 0.163 | 0.119 | 0.272 | 1.20 | – |
| 10 000 | 41.0 | 0.647 | 0.126 | 0.094 | 0.125 | 0.041 | 0.241 | 1.31 | 33.9 |
| 30 000 | 13.68 | 0.516 | 0.102 | 0.071 | 0.088 | 0.007 | 0.191 | 1.13 | 4.0 |
| 70 000 | 11.10 | 0.471 | 0.079 | 0.060 | 0.067 | 0.004 | 0.167 | 1.06 | 2.1 |
| **130 000** | **9.99** | **0.449** | **0.062** | **0.052** | **0.058** | **0.002** | **0.152** | 1.02 | 1.4 |

The generator loss spikes at the GAN start and falls as the critics saturate. No NaNs in any
of the 2 600 records.

### Validation (32 clips, free-running)

| metric | 12 000 | 130 000 |
| --- | --- | --- |
| `predicted_mel` | 0.787 | 0.805 |
| `teacher_forced_mel` | – | 0.755 |
| `pitch_error_semitones` | 2.96 | 3.18 |
| `pitch_spread_ratio` | – | 0.891 |
| `voicing_accuracy` | 0.929 | 0.926 |
| `pause_accuracy` | 0.946 | 0.948 |
| `duration_length_ratio` | 1.063 | 1.051 |

`predicted_mel` rises slightly while the listening metrics improve: the GAN trades mean
mel error for sharper high-frequency structure. Trust the ASR/CER numbers and a listening
test over the mel loss late in training.

## Checkpoints

`runs/full/` (1.25 GB, 19 files):

| file | contents |
| --- | --- |
| `inference.pt` | generator only, 17.4 MB — what `infer`/`evaluate`/`synthesize_samples` read |
| `last.pt` | full state (weights, critics, both optimizers, sampler, RNG), 83 MB |
| `step-00008000.pt` | GAN start, 62.7 MB (no critics yet) |
| `step-00010000.pt … step-00130000.pt` | every 10 000 steps, 83.3 MB |
| `config.json`, `recipe.json`, `metrics.jsonl` | run provenance |

`scripts/checkpoint_probe.py --run runs/full` synthesises one fixed sentence from each
checkpoint so you can hear the trajectory.

## Results

| | |
| --- | --- |
| Held-out test (6 clips, free-running) | CER **11.8 %** (reference 2.6 %), WER 20.8 % (4.2 %), duration ratio 0.907, pause ratio 0.956, mel DTW 0.778, pitch std 6.09 vs 5.78 st |
| Phase 8 samples (7 utterances) | mean CER **15.7 %**, WER 27.5 %, **8.4 %** with numerals folded; 3 transcribed exactly |
| Front-end consistency | `frontend_token_mismatches: []` |

Details and the per-clip table: `GUIDE.md`, `FINAL_REPORT.md`.

## Tuning

| goal | change |
| --- | --- |
| more VRAM headroom / faster steps | `--batch-frames 8000 --max-batch 32` (~11 GB) |
| longer training | raise `--steps` and `--lr-half-life` together, resume |
| less overfitting on a small corpus | lower `--gan-ratio` (0.5), shorter `--gan-ramp` |
| faster GAN convergence | `--gan-start 5000` (the teacher-forced phase needs ~3 000 steps to stabilise durations) |
| new sentences / new speaker | add rows to the Parquet, re-run prepare + annotate, resume from `last.pt` (`--new-data` when the split set changes) |
| compare recipes on the same data | `--init-from runs/full/last.pt --output runs/branch` (branches the GAN phase, keeps trained critics) |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| OOM inside the GAN phase | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then lower `--batch-frames` |
| `checkpoint already reached requested total steps` | `--steps` ≤ the checkpoint's step; raise it |
| `resume inputs … changed` from `prepare` | the shard was rewritten; re-prepare that split |
| training is slow at the start of a chunk | the first ~50 steps warm the CUDA caches and the sampler pool |
| no audio in the run directory | `train` writes no audio by design; use `synthesize_samples.py` / `checkpoint_probe.py` |
| log looks like `s t e p` with gaps | UTF-16LE from `*>>`; `parse_train_log.py` strips the NULs |
