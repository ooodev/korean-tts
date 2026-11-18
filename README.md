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
| Demo | [**Browser demo**](https://ooodev.github.io/korean-tts/) — the model runs in the page (ONNX + Pyodide), nothing leaves the device |

## Try it

In the browser: **[ooodev.github.io/korean-tts](https://ooodev.github.io/korean-tts/)** —
type a sentence, press 합성. About 40 MB downloads once; a second visit starts from cache.

From the command line:

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
| [`tiny-voca/web/README.md`](tiny-voca/web/README.md) | the browser demo: how it loads the model, the assets, the gates, deploying it |

## Layout

```
samples/            final 24 kHz audio (7 utterances) + token and ASR tables
runs/full/          inference.pt is the model; config, recipe, metrics.jsonl
annotations_full/   the aligned clips the trainer read (32 MB)
weights/phonemizer/ dizione's English dictionaries
eval/full/          report.json + generated/reference audio pairs
scripts/            workspace helpers (prepare, split, merge, train driver, checks)
tiny-voca/          the model, with the Korean front end
  web/public/       the browser demo: page, worker, and the assets Pages serves
dizione/            the English side of the front end (upstream, unmodified)
config-ko.json      24 kHz · hop 256 · 80 mels · speakers ["KSS"] · vocab_size 142
```

`tiny-voca/` and `dizione/` are vendored as plain files: `tiny-voca/` carries the Korean
front end on top of upstream, `dizione/` is unmodified.

The repository is 102 MB and every file is under 25 MB, so no Git LFS is needed. The large
regenerable layers are the ones a clone does not carry: `data/` (1.9 GB Parquet),
`features2/` (5.7 GB acoustic features) and the 83 MB resume checkpoints `runs/full/last.pt`
and `step-*.pt`. `weights/` also holds the 1.2 GB aligner and Whisper checkpoints, which are
downloaded on first use.

## Further training — growing KSS's style repertoire

(Annotated on the v2 checkpoint `runs/v2/inference.pt` — 4.55 M params, 3 speakers, 18.13 h of
data, 160 000 steps; the table above documents the v1 Korean-only baseline. v2 details:
`FINAL_REPORT_v2.md`.)

The model has absorbed what its data offers: on the Korean gate, human KSS audio itself scores
5.70 % CER with the same ASR, and the model scores 5.35 % — the metric sits at the
recognizer's noise floor. More steps or a larger model will not move this. What remains is
**style**: things the training data never showed the model. The efficient path is new KSS data
covering those styles, plus continued training done the way below.

### Which styles to grow (in priority order)

| Style | Where it lives today | Cheapest improvement |
| --- | --- | --- |
| Korean numerals — dates, amounts, temperatures, populations | text normalization + existing KSS rows (a strength) | optional 30–60 min of KSS numeral sentences; judge by ear — CER can't see this |
| English words spoken Korean-style (Python → 파이썬), every English span | `data/loanwords.tsv` (462 entries) under Mode A; unknown words fall back | **grow the dictionary first — zero training cost**; then record code-mixed sentences for real co-articulation |
| Special characters & punctuation — read naturally or skipped | front-end normalization handles the common cases | add punctuation-heavy sentences to any new pool; gate on 0 `<unk>` before training |
| Emotion & expression | weakest dimension — the KSS corpus is one calm style; prosody layers + pitch/energy conditioning exist | only real recordings in varied styles teach it; voice conversion can't fake breathing or micro-dynamics |

### Get the data (best → worst)

1. **Real KSS recording** — the only route that keeps the timbre pure. Calibrate to F2's
   5.07 h: record **~5 h of KSS English** (3–4 h English + ~1 h Korean–English mixed +
   30–60 min numerals), 24 kHz mono, clean and normalized. New recordings need the speaker's
   consent and keep the CC BY-NC-SA chain.
2. **Voice conversion** (F2/M2 English → KSS timbre) — a workable fallback; expect fine-prosody
   artifacts.
3. **Relabelling F2 rows as KSS — do not.** Two timbres cannot share one speaker embedding; the
   Korean output drifts toward a mix. This is the classic multi-speaker failure mode.

### Train it without wasting compute

- Branch with `train --init-from runs/v2/inference.pt --new-data <merged annotations>` — it
  inherits weights, step counter and GAN critics. `--resume` is for the identical data pool only.
- The learning rate keeps halving on its own (160 k ≈ ⅓ of base, 200 k = ¼ of base). The low LR
  is exactly what protects the existing abilities — don't reheat.
- The old data replays automatically: the sampler draws uniformly from the merged pool, so the
  old 16 451 clips still receive ≈85 % of every step.
- **Mind the pass count.** With the v2 recipe (12 000 frames/batch = 128 s of audio), the old
  pool was trained for ≈315 passes. In +40 k steps on the merged pool, each clip gets ≈65–70
  more draws — so a new ~3 000-clip pool ends at ≈70 cumulative passes versus ≈380 for the old
  abilities, about 1/5 of the exposure. Make up the difference, in order of ROI:
  1. **up-sample the new clips 3–5× in the manifest** (duplicate rows before merging) →
     ≈150–220 passes at the same wall time;
  2. train further (160 k → 250 k ≈ +7 h);
  3. reheat the LR — last resort, it shakes the old gate.
- Tailwind: the acoustic side already knows English from F2/M2 (≈300 passes). KSS English is
  speaker-adaptation territory — the model learns the (KSS, English) mapping and the embedding,
  not English from scratch — so it converges faster than it looks.

### Gate before you keep it

| Check | Threshold |
| --- | --- |
| KO CER (228-sentence test split) | 5.35 % ± 0.3 % — don't regress, don't chase (ASR floor ≈ 5.7 %) |
| EN WER on KSS-English rows | ≤ 1.5 % |
| pause_ratio | 1.16 → toward 1.0 (the one measurable "extra pauses" trait) |
| duration_ratio / pitch_std | ≈0.975 / ≈5.0 st — no speech-rate or intonation drift |
| Human listening battery | numeral / loanword / punctuation / emotion sentences — ears decide timbre and prosody |

`--init-from` starts a new run, so `runs/v2` stays intact for rollback.

### One routing decision before recording

KSS currently reads English spans through Mode A (loanword G2P: Python → 파이썬). If the new
recordings speak English **natively**, add a mode that routes English spans to dizione —
otherwise the model learns what the front end refuses to render. "파이썬 or paɪθɑːn" is a
product decision; the training follows.

### Capacity reality check

4.55 M params is intentionally small (the browser demo ships a ~21 MB ONNX). It scales with a
few hours of data per speaker — which is what this recipe buys. A 7 000 h multi-speaker corpus
belongs to a larger model: at that scale the extra hours mostly mean "many voices", and 4.5 M
cannot hold them.

KSS is licensed **CC BY-NC-SA 4.0** — non-commercial research/testing only.
