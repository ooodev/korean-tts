# GUIDE.md ??what was produced and how to test it

Start here if you just want to listen and try the model.
Background: `AGENTS.md` (working in this repo), `DATASET.md` (data), `TRAIN.md` (training),
`FINAL_REPORT.md` (full results), `WORKLOG.md` (decision log), `guide.html` (same content as
a browsable page with inline audio players), `tiny-voca/web/README.md` (the browser demo).

All paths are relative to `C:\ai-workspace\korean-tts`.

## Deliverables

```
samples/                        ??final audio (24 kHz mono)
  sample_01_greeting.wav            4.11 s   385 KB   ?덈뀞?섏꽭?? ?ㅻ뒛 ?좎뵪媛 ?뺣쭚 留묎퀬 醫뗫꽕??
  sample_02_question.wav            3.67 s   344 KB   吏湲?紐??쒖씤吏 ?뱀떆 ?뚮젮二쇱떎 ???덈굹??
  sample_03_phonology.wav           4.67 s   438 KB   援?Ъ 留쏆씠 ?앸궡二쇨퀬 ?좊씪????궗???λ?濡?뒿?덈떎.
  sample_04_numbers.wav             4.70 s   441 KB   ?댁쿇?댁떗??????씪 ????삤 ???ㅽ썑 ?????쇱떗 遺꾩엯?덈떎.
  sample_05_long.wav                6.22 s   583 KB   ???蹂댄넻 ?댁떗 遺??뺣룄 ??옞???먭퀬 ?섎㈃ ??  sample_06_exclamation.wav         6.10 s   572 KB   ?, ?뺣쭚 ??⑦븯?ㅼ슂! ?쒕뵒?????깃났?덉뒿?덈떎!
  sample_07_codemix_check.wav       3.61 s   338 KB   ?ㅻ뒛 Python?쇰줈 ?뚯뒪?몃? 吏꾪뻾?덉뒿?덈떎.
  manifest.json                     the 7 texts
  tokens.json                       tokens synthesised per clip + length + peak
  asr_check.json                    ASR back-transcription table (CER / WER)

runs/full/                      ??model (1.25 GB)
  inference.pt                      17.4 MB ??what you use for inference (4 549 579 params)
  last.pt                           resume point, step 130 000
  step-00008000.pt ??step-00130000.pt   GAN start + every 10 000 steps
  config.json  recipe.json  metrics.jsonl

eval/full/                      ??evaluation
  report.json                       CER/WER, duration, pitch, DTW, per-clip records
  audio/kss-XXXXX-generated.wav     6 synthesised clips
  audio/kss-XXXXX-reference.wav     the same sentences as recorded (A/B comparison)
  audio/novel0{0,1}_KSS-generated.wav  two sentences from no split

eval/precheck/                  ??step 80 000 mid-training check (same 7 sentences)
tiny-voca/web/public/           ??browser demo (published to GitHub Pages)
  index.html  worker.js           Korean page + Pyodide/ONNX worker (en.html = the English one)
  assets/                         model.onnx 21.4 MB, dictionaries, pyfront.zip, dizione.zip
data/  features2/  annotations_full/   ??Parquet, acoustic features, alignments
FINAL_REPORT.md  WORKLOG.md  guide.html  tiny-voca/web/README.md
logs_train_full.txt             ??training log, 5.4 MB JSONL
```

## 1. Listen

```powershell
explorer C:\ai-workspace\korean-tts\samples
Invoke-Item C:\ai-workspace\korean-tts\samples\sample_01_greeting.wav

# generated vs recorded, same sentence
Invoke-Item C:\ai-workspace\korean-tts\eval\full\audio\kss-00512-generated.wav
Invoke-Item C:\ai-workspace\korean-tts\eval\full\audio\kss-00512-reference.wav
```

`guide.html` has the same files as inline players, so you can A/B without leaving the page.

## 2. Synthesise your own sentence

Set the environment once per shell, then one command per sentence (~5 s for ~4.7 s of audio):

```powershell
cd C:\ai-workspace\korean-tts
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko"
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"

.\.venv\Scripts\python.exe -m tiny_voca.infer `
  --checkpoint runs\full\inference.pt `
  --text "?댁씪 ?꾩묠 ?댁떆???뚯쓽???대갚?먯꽌 留뚮굹湲곕줈 ?덉뼱??" `
  --output samples\my_test.wav --device cuda
```

| flag | effect |
| --- | --- |
| `--speed 0.9` | whole utterance 10 % slower |
| `--pitch-shift 0.058` | adds to log F0 ??one semitone up |
| `--pitch-range 1.2` | scales pitch movement around the utterance mean |
| `--voice KSS` | voice name (this model has the single speaker `KSS`) |
| `--frontend ko` | only needed if `TINY_VOCA_FRONTEND` is unset |

Check the result:

```powershell
.\.venv\Scripts\python.exe scripts\verify_wavs.py samples\my_test.wav
.\.venv\Scripts\python.exe scripts\asr_check.py samples\my_test.wav `
  "?댁씪 ?꾩묠 ?댁떆???뚯쓽???대갚?먯꽌 留뚮굹湲곕줈 ?덉뼱??" --device cuda
```

## 3. The browser demo

**[ooodev.github.io/korean-tts](https://ooodev.github.io/korean-tts/)** runs the same model
in the page: ONNX Runtime Web for the graph, Pyodide for the phonemizer, so the text never
leaves the device. Serve it locally with

```powershell
.\.venv\Scripts\python.exe -m http.server 8081 --directory tiny-voca\web\public  # http://127.0.0.1:8081/
```

The assets are in the repository (`tiny-voca/web/public/assets`, 28 MB) and are rebuilt with
`scripts/build_web_assets.py`; `tiny-voca/web/README.md` explains the pieces. Two gates cover
it: `scripts/check_onnx.py` (the graph speaks like the checkpoint) and `scripts/check_shim.py`
(the pure-Python MeCab stand-in, which Pyodide needs, reproduces the front end on held-out
sentences).

| | |
| --- | --- |
| Model | `model.onnx` 21.4 MB, exported from `runs/full/inference.pt` |
| Phonemizer | `pyfront.zip` ??g2pK + jamo + `kofront` (the same `vocab/text/korean.py`) + dizione |
| Morphology | `mecab-dict.json` ??20 649 corpus eojeols with their real MeCab analyses |
| Measured in Chrome | published site: 4.7 s of audio in 0.28 s (16.9× real time); locally 3.6 s in 0.37 s (9.7횞 real time), phonemization 186 ms |
| First visit | ~40 MB, then from Cache Storage |

## 4. Re-run the gates

```powershell
# audio: 24 kHz, mono, duration, RMS, peak, NaN/Inf
.\.venv\Scripts\python.exe scripts\verify_wavs.py "samples/*.wav"

# intelligibility: faster-whisper small (ko) back-transcription ??CER / WER
.\.venv\Scripts\python.exe scripts\asr_check.py --manifest samples\manifest.json `
  --outdir samples --model small --device cuda --json samples\asr_check.json

# held-out evaluation, free-running synthesis
.\.venv\Scripts\python.exe -m tiny_voca.evaluate --checkpoint runs\full\inference.pt `
  --annotations annotations_full --output eval\full --tag full `
  --splits test --per-voice 6 --novel 2 --device cuda

# training curve
.\.venv\Scripts\python.exe scripts\parse_train_log.py logs_train_full.txt --every 10000 --validation

# front-end + aligner unit tests
cd tiny-voca; ..\.venv\Scripts\python.exe -m unittest discover -s tests; cd ..   # 67 OK, 2 skipped

# rebuild the HTML page from the artefacts
.\.venv\Scripts\python.exe scripts\make_guide.py
.\.venv\Scripts\python.exe scripts\check_guide.py
```

## 5. Final samples ??measured

CER is character-level, spaces ignored. The last column folds a digit run and a run of
Sino-Korean numeral morphemes to one placeholder, because Whisper writes "?댁쿇?댁떗???? as
"2026?? ??that column measures the reading, not the orthography.

| file | text | dur | RMS | peak | CER | WER | numbers folded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 greeting | ?덈뀞?섏꽭?? ?ㅻ뒛 ?좎뵪媛 ?뺣쭚 留묎퀬 醫뗫꽕?? | 4.11 s | 0.087 | 0.513 | **0.0 %** | 0.0 % | 0.0 % |
| 02 question | 吏湲?紐??쒖씤吏 ?뱀떆 ?뚮젮二쇱떎 ???덈굹?? | 3.67 s | 0.079 | 0.627 | **0.0 %** | 0.0 % | 0.0 % |
| 03 phonology | 援?Ъ 留쏆씠 ?앸궡二쇨퀬 ?좊씪????궗???λ?濡?뒿?덈떎. | 4.67 s | 0.090 | 0.524 | 10.0 % | 50.0 % | 10.0 % |
| 04 numbers | ?댁쿇?댁떗??????씪 ????삤 ????| 4.70 s | 0.081 | 0.544 | 54.5 % | 90.9 % | **12.5 %** |
| 05 long | ???蹂댄넻 ?댁떗 遺??뺣룄 ??옞???먭퀬 ?섎㈃ ??| 6.22 s | 0.081 | 0.590 | 15.6 % | 26.7 % | **6.5 %** |
| 06 exclamation | ?, ?뺣쭚 ??⑦븯?ㅼ슂! ?쒕뵒?????깃났?덉뒿?덈떎! | 6.10 s | 0.084 | 0.583 | **0.0 %** | 0.0 % | 0.0 % |
| 07 code-mix | ?ㅻ뒛 Python?쇰줈 ?뚯뒪?몃? 吏꾪뻾?덉뒿?덈떎. | 3.61 s | 0.095 | 0.566 | 30.0 % | 25.0 % | 30.0 % |
| **mean** | | 4.73 s | | | **15.7 %** | **27.5 %** | **8.4 %** |

* **03** checks the phonological rules: nasalization 援?Ъ, liquidization ?좊씪 ??[?ㅻ씪],
  tensification ?앸궡二?, liaison ??궗?? They survive into the audio; the residual is
  Whisper's eojeol spacing plus one syllable of ?λ?濡?뒿?덈떎.
* **04** is spelled correctly in the audio; the 54.5 % is a transcription convention.
* **07** is the honest weak spot: `Python` is phonemized by dizione and attached to ?쇰줈
  with no inserted space, but KSS has no English recordings, so those phones are unseen.

## 6. Held-out evaluation

`tiny_voca.evaluate` feeds text only and predicts durations, pitch, voicing and energy
(free-running). 6 test sentences, both generated and recorded audio transcribed by the same
ASR model.

| metric | generated | recorded |
| --- | --- | --- |
| CER | **11.8 %** | 2.6 % |
| WER | 20.8 % | 4.2 % |
| duration ratio | 0.907 | ??|
| pause ratio | 0.956 | ??|
| DTW log-mel | 0.778 | ??|
| pitch std (semitones) | 6.09 | 5.78 |
| voiced fraction | 0.673 | 0.572 |

Three of six are transcribed exactly; `frontend_token_mismatches: []` means re-running the
front end on the raw text reproduces the annotated tokens. The reference column is the ASR
model's own floor.

## 7. Scripts

| script | purpose |
| --- | --- |
| `prepare_kss.py` | HF KSS ??24 kHz Parquet with the columns `prepare` expects |
| `resplit_kss.py` | group-wise (sentence) re-split, seed 42 |
| `split_shards.py` / `merge_features.py` | parallel acoustic preparation and its merge |
| `train_full.ps1` | chunked training driver with `--resume` retry loop |
| `parse_train_log.py` | instantaneous steps/s, audio-s/s, losses, validation from a log |
| `synthesize_samples.py` | manifest ??`samples/*.wav` + `tokens.json` |
| `verify_wavs.py` | audio gate (accepts a glob) |
| `asr_check.py` | Korean back-transcription, CER/WER + numeral folding |
| `check_units.py` | aligner unit-window invariants (8 cases) |
| `checkpoint_probe.py` | one fixed sentence from every checkpoint in a run |
| `verify_phase5_align.py` | 20-clip alignment gate |
| `build_mecab_dict.py` | MeCab analyses of the training split ??`mecab-dict.json` for the browser |
| `build_web_assets.py` | assemble `tiny-voca/web/public/assets` (`--export` re-runs the ONNX export) |
| `check_onnx.py` | ONNX graph vs the checkpoint (durations, correlation, exact-noise graph) |
| `check_shim.py` | pure-Python MeCab stand-in vs the real one, on held-out sentences |
| `make_guide.py` / `check_guide.py` | build and validate `guide.html` |

## 8. Fresh environment

```powershell
git clone https://github.com/ooodev/korean-tts.git   # carries tiny-voca/ and dizione/
cd korean-tts
uv venv .venv --python 3.11
$env:VIRTUAL_ENV="C:\ai-workspace\korean-tts\.venv"
uv pip install -e ./dizione -e ./tiny-voca torch --index-url https://download.pytorch.org/whl/cu126
uv pip install datasets pyarrow soundfile librosa torchaudio scipy g2pk transformers `
               faster-whisper jiwer
.\.venv\Scripts\python.exe -c "import dizione; print(dizione.phonemize('Hello, world!'))"
.\.venv\Scripts\python.exe -c "from g2pk import G2p; print(G2p()('留묒? ?좎뵪媛 醫뗭븘??'))"   # 留먭렐 ?좎뵪媛 議곗븘??
```

`weights/phonemizer/` (dizione's `lexicon.tsv` + `g2p.npz`) is in the clone, so
`DIZIONE_DATA` works straight away; the aligner and Whisper checkpoints download on first
use. With the clone you can synthesise immediately ??`runs/full/inference.pt` is committed.
To retrain, follow the eight stages in `AGENTS.md`, or the reproduce block in
`FINAL_REPORT.md`; `data/` and `features2/` are the layers you have to rebuild.

## 9. Known limits

* **English accent** ??the 68 English phones are unseen in KSS; the front end is correct,
  the accent is not learned.
* **Single speaker** ??`KSS` only.
* **CER inflation** ??read the "numbers folded" column for sentences with numerals.
* **Metrics support listening, they do not replace it** ??judge by `samples/` and the
  `eval/full/audio/` A/B pairs.
* **Licence** ??KSS is CC BY-NC-SA 4.0 (non-commercial research/testing).
