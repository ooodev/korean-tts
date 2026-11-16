# Korean & code-mixed TTS on tiny-voca + dizione

A 24 kHz Korean text-to-speech model with a Korean + English code-mixed front end, trained
on [`Bingsu/KSS_Dataset`](https://huggingface.co/datasets/Bingsu/KSS_Dataset) on top of
[`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca) and
[`n4ze3m/dizione`](https://github.com/n4ze3m/dizione).

| | |
| --- | --- |
| Front end | 142-token unified vocabulary — dizione English (ids 0–75, stress intact) + Korean NFD Jamo; g2pk phonology, silent `ㅇ` stripped |
| Corpus | 12 854 clips → **8 504 aligned** (8.45 h) at 24 kHz, 0 `<unk>`, sentence-wise 95/2.5/2.5 split |
| Aligner | `wav2vec2-large-xlsr-korean` CTC Viterbi at syllable level, sub-allocated to Jamo by energy/voicing |
| Model | 4 549 579 parameters, 130 000 steps (GAN from 8 000), 11 h 35 min on an RTX 4090 |
| Held-out test | **CER 11.8 %** (recorded reference 2.6 %), duration ratio 0.907, pitch spread 6.09 vs 5.78 st |
| Samples | 7 utterances, **mean CER 15.7 %** (8.4 % with numerals folded), 3 transcribed exactly |

## Try it

```powershell
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko"
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"

.\.venv\Scripts\python.exe -m tiny_voca.infer --checkpoint runs\full\inference.pt `
  --text "내일 아침 열시에 회의실 이백에서 만나기로 했어요." --output samples\my_test.wav --device cuda
```

## Documentation

| document | about |
| --- | --- |
| [`GUIDE.md`](GUIDE.md) | what was produced, listening, synthesising your own sentences, re-running the gates |
| [`TRAIN.md`](TRAIN.md) | model config, recipe, throughput/VRAM, monitoring, checkpoints, tuning |
| [`DATASET.md`](DATASET.md) | KSS → Parquet → features → alignments, schemas, splits, QC gates |
| [`AGENTS.md`](AGENTS.md) | working in this repository: environment, env vars, gotchas, stage order |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | full results: files changed, vocab, aligner, training, evaluation, samples |
| [`WORKLOG.md`](WORKLOG.md) | the running log: every gate output, command, timing and decision |
| [`guide.html`](guide.html) | the guide as a page with inline audio players |

## Layout

```
samples/            final 24 kHz audio (7 utterances) + token and ASR tables
runs/full/          inference.pt is the model; config, recipe, metrics.jsonl
annotations_full/   the aligned clips the trainer read (32 MB)
weights/phonemizer/ dizione's English dictionaries
eval/full/          report.json + generated/reference audio pairs
scripts/            workspace helpers (prepare, split, merge, train driver, checks)
tiny-voca/          the model, with the Korean front end
dizione/            the English side of the front end (upstream, unmodified)
config-ko.json      24 kHz · hop 256 · 80 mels · speakers ["KSS"] · vocab_size 142
```

`tiny-voca/` and `dizione/` are vendored as plain files: `tiny-voca/` carries the Korean
front end on top of upstream, `dizione/` is unmodified.

The repository is 75 MB and every file is under 25 MB, so no Git LFS is needed. The large
regenerable layers are the ones a clone does not carry: `data/` (1.9 GB Parquet),
`features2/` (5.7 GB acoustic features) and the 83 MB resume checkpoints `runs/full/last.pt`
and `step-*.pt`. `weights/` also holds the 1.2 GB aligner and Whisper checkpoints, which are
downloaded on first use.

KSS is licensed **CC BY-NC-SA 4.0** — non-commercial research/testing only.
