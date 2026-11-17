# V2_SPEC.md — bilingual TTS v2.0 (Korean + English, one 4.5M model)

Locked 2026-11-17. This document is the engineering contract for v2.0: one
multilingual model that speaks Korean natively (≥ v1.0 quality), reads English,
and pronounces Arabic numerals the Korean way. Top priority: **front-end quality
before GPU time** — a clean, unambiguous, verifiable phoneme stream is the gate
to alignment and training.

## 0. Serving modes (one checkpoint, two front ends)

| Mode | Front end | English spans | Numbers | Voice |
| --- | --- | --- | --- | --- |
| **A — "Korean speaker"** (default) | `ko-loan` (new) | loanword G2P → Jamo | Korean TN | KSS |
| **B — "native English"** | `own` (existing) | dizione IPA (stress 0/1/2) | English | F2 / M2 |
| C — (reserved, v2.1) | `ko` (existing) | dizione IPA under KSS voice | — | KSS |

Mode A is fully Korean tokens (g2pk Jamo + loanword Jamo + TN Jamo), so a Mode A
sentence is spoken by the model's Korean pathway — no cross-language acoustic
interpolation is required at inference for the core use case.

## 1. Decision log

| # | Decision | Rationale | Fallback / trigger |
| --- | --- | --- | --- |
| D1 | Train v2.0 **strictly from scratch**; v1.0 (`runs/full/inference.pt`) stays untouched as baseline (workspace is a deliberate clean slate) | Clean language balance across the 4.5M budget; no converged-basin priors; the resumable `last.pt` was intentionally deleted, so a warm start is not a free option anyway | If Korean CER > 12.5 % (v1.0: 11.8 %) at the 120k checkpoint → warm-start branch: a ~50-line script rebuilds a training snapshot from `inference.pt` (generator weights + `step=130000` + fresh critics/optimizer/sampler), then `--init-from --new-data` (machinery already built in `train.py`; the GAN gradient scaling tolerates fresh critics) |
| D2 | **Keep the 142-token inventory** — no IPA redesign | `EN_SYMBOLS` (0–75, IPA) and `KOREAN_PHONES` (76–141, Jamo) are disjoint (`vocab.py:1-9`): zero cross-language collisions already. 3-way stop contrast ㅂ/ㅃ/ㅍ, ㄷ/ㄸ/ㅌ, ㄱ/ㄲ/ㅋ are separate rows; ㅓ/ㅡ/ə/ʌ separate. Shared IPA would break the g2pk contract (it emits Jamo) and lose the contrast | — |
| D3 | Korean-accented English = **loanword G2P to Jamo**, not IPA substitution | (1) Reuses the model's proven Korean acoustic pathway — zero new acoustic learning, so it is the *highest-clarity* route; (2) matches how real Korean speakers read English (헬로, 스마트폰); (3) IPA substitution (θ→s, ð→d on English rows) yields "English with odd consonants" that no Korean speaks and the model has never heard — worse clarity, not better | If loanword coverage is limiting on real prompts → grow the curated dictionary (fallback IPA must stay < 2 % of prompt tokens) |
| D4 | **Dual aligners** for v2.0 (kalign for KO incl. loanwords, espeak-CTC for EN); unified CTC aligner deferred | The specialized aligners are proven (v1.0 gates, 33 clips/s, tuned QC). Loanword output carries Hangul syllable `units`, so kalign aligns Mode A text directly. A unified aligner is only needed once cross-language acoustic data enters | If synthetic code-switched audio is adopted (v2.1) → train one wav2vec2-CTC over the 142 inventory bootstrapped from existing per-language alignments, with a parity gate vs. the specialized aligners on held-out pure clips before any switch |
| D5 | **No language embedding** in v2.0 | Jamo and IPA token sets are disjoint → the language of every spoken token is already unambiguously encoded in the token stream; a redundant embedding spends capacity in a tight 4.5M budget | If validation shows cross-language prosody bleed (English rhythm inside Korean sentences) → add a per-token 2-class language embedding fed from front-end span segmentation in v2.1 |
| D6 | v2.0 corpus = **KSS (KO) + n4ze3m/tiny-voca (EN)** only; no synthetic code-switched or number-rich data in v2.0 | With D3, Mode A needs no cross-language audio. KSS transcripts already carry numbers written out as spoken Korean (12.8k sentences) | If number/loanword evaluation is weak → targeted synthetic data (Qwen3-TTS with KSS reference) in v2.1, which would activate D4's unified aligner |

## 2. Front-end spec (Phase 1 — Phoneme Project)

### 2.1 New mode `ko-loan` (Mode A)
- Korean spans → `g2pk` (all sandhi rules: liaison, tensification, nasalization,
  lateralization, palatalization, ㅎ-aspiration/deletion, final-consonant
  neutralization) — already implemented via `korean_pronounce()`.
- English spans → loanword G2P (§2.3) → Jamo tokens, stress 0, syllable `units`.
- Digits/symbols → Korean TN (§2.2).
- `own` (Mode B) and `ko` (Mode C) front ends are byte-unchanged; their existing
  tests must stay green.

### 2.2 Korean TN rules (Mode A) — final reading table (2026-11-17, user sign-off pending)
Implementation: rework `expand_numbers` + counter/symbol tables in `korean.py`; in Mode A
this runs **before** `segment()`, so digits/symbols never reach `EN_SPAN` or the aligner.
Rule sources (MIT/Apache-2.0, rules extracted, no dependency added): g2pK `numerals.py`
(counter list, native/Sino readings), MB-iSTFT-VITS-Korean `text/korean.py` (same list,
`latin_to_hangul`, `fix_g2pk2_error`), MeloTTS `korean.py`/`ko_dictionary.py`
(dictionary-first normalization pattern), CoreaSpeech N2gk+ `normalization.py` (unit
categories, phone patterns, symbol map, pipeline order).

| Input | Reading | Note |
| --- | --- | --- |
| `3개 3명 3살 3마리 3장 3통 3팩 3권 3잔 3끼 3칸 3번 3시 3방 3수 3술 3짝 3채 3척 …` (n ≤ 20, g2pK `BOUND_NOUNS` **minus `세`/`박`**) | `세개 세명 …` | **Fixes current bug**; **user-corrected**: `3세` (age) → `삼 세`, `3박 4일` → `삼 박 사 일` (Sino); `번` native approved |
| `3층 3년 3월 3일 3분 3초 3원 3회 3등 3점 3호 3기 3차 3페이지 3코어 …` (Sino units) | `삼층 …` | Sino reading (current behavior, keep) |
| `3시 30분` | `세시 삼십분` | keep |
| `14:30`, `13:15`, `20:00`, `0시` | `십사시 삼십분`, `십삼시 십오분`, `이십시`, `영 시` | **user-corrected**: hours 1–12 native (`한 시`…`열두 시`), 0 → `영 시`, 13–24 Sino |
| `2026년 3월 14일` | `이천이십육년 삼월 십사일` | keep |
| `6월`, `10월` | `유월`, `시월` | N2gk exception — flagged for review |
| `20살` | `스무 살` | N2gk exception (idiom) |
| `010-1234-5678`, `01012345678` | `공일공 일이삼사 오육칠팔` | **Fixes current bug**: phone pattern runs first; hyphen → pause; leading 0 → 공 |
| `123번`, `402호` (≥ 3 digits + 번/호) | `일이삼번`, `사공이호` | digit-by-digit, **0 → 공** (never 영); full Sino `사백이호` also acceptable |
| `3번째` | `세 번째` | ordinal → native numeral + 번째 |
| `5%` | `오 퍼센트` | keep |
| `$12.99` | `십이 달러 구십구 센트` | **user-confirmed**; cents as a number |
| `$12` | `십이 달러` | **user-confirmed**; integer currency, no cents |
| `₩3000`, `€20`, `£5`, `¥100` | `삼천 원`, `이십 유로`, `오 파운드`, `백 엔` | currency symbol → Sino numeral + unit |
| `3/14` (2 parts, M ≤ 12 and D ≤ 31) | `삼월 십사일` | date reading is the default |
| `14/3` (M > 12) | `삼 분의 십사` | fraction: denominator 분의 numerator |
| `2026/3/14`, `2026-03-14` | `이천이십육년 삼월 십사일` | 3-part date |
| `100~200명` | `백에서 이백 명` | range (N2gk) |
| `1+1` | `원플러스원` | etc.-dictionary pattern (MeloTTS) |
| `3.5`, `3.05` | `삼점오`, `삼점영오` | decimal; 0 inside the decimal part → 영 |
| `123` bare | `백이십삼` | keep |
| `10km`, `2kg`, `500ml`, `60°C` | `십 킬로미터`, `이 킬로그램`, `오백 밀리리터`, `육십 도` | unit table (N2gk): km→킬로미터, kg→킬로그램, ml→밀리리터, ℃/°C/°→도 |
| `19세기`, `20세기` | `십구세기`, `이십세기` | centuries → Sino |

Digit-run heuristic (digit-by-digit when triggered): phone pattern (3-3/4-4 or 11 digits),
≥ 3 digits followed by 번/호, ≥ 5 consecutive digits, code-like `X\d+`. Otherwise → Sino
with the counter table. **0 → 공 in every digit-by-digit run** (phones/codes/room numbers;
never 영); 영 only for bare 0 and inside decimal parts (`3.05 → 삼점영오`).

### 2.3 Loanword G2P pipeline (Mode A)
1. Normalize: casefold, strip punctuation.
2. **Dictionary lookup**: curated TSV of high-frequency loanwords + tech/brand names
   (`tiny_voca/data/loanwords.tsv`, **LLM-drafted seed, user review pending — Q1 resolved**;
   overridable via `TINY_VOCA_LOANWORDS` env var for the growth loop). Hit → exact Hangul.
3. **Acronym spelling**: all-caps runs of length 2–6 (`USB`, `AI`, `API`, `CEO`) →
   letter names (에이아이, 유에스비, 씨이오) via the MeloTTS/N2gk+ letter map (R=알).
4. **OOV fallback** (deterministic, no leaks/crashes): dizione IPA (76-symbol inventory)
   → IPA→Jamo state machine ported from g2pK `convert_eng` (외래어 표기법 rules 1–8:
   PTK coda/으 logic, BDG+으, fricative+으, ʃ→시/슈/샤…시, tʃ/dʒ→치/지, nasal codas,
   L→ㄹㄹ, w/j+vowel syllable merging) → syllabification (7 legal codas, cluster break
   with ㄱ+으).
5. Syllabify → Hangul → NFD Jamo tokens, stress 0, `units` = Hangul syllables
   (**kalign-compatible** — the reason D4 holds).
6. Gate: dictionary + acronym stages keep the IPA fallback < 2 % of prompt tokens,
   else dictionary/rules must be extended.

### 2.4 Front-end verification gate (must pass before any alignment)
- New `tests/test_frontend_gate.py`, ≥ 120 golden cases with expected token
  sequences: Korean sandhi (20), TN (40), loanword (30), Mode A segmentation
  (20), Mode B regression (10). 100 % pass required.
- Zero `<unk>` on every golden prompt; `frontend_token_mismatches: []` discipline
  from v1.0 preserved.
- Existing 67-test suite stays green.

## 3. Data spec (Phase 2)

| Corpus | Hours (train) | Aligner | QC limits | Screening |
| --- | --- | --- | --- | --- |
| KSS (existing `annotations_full/`) | 8.448 | kalign (done) | 0.6 / 0.3 / 90 | done (v1.0 gates) |
| n4ze3m/tiny-voca EN | ~17.5 | espeak-CTC (`facebook/wav2vec2-lv-60-espeak-cv-ft`, one-time ~1.3 GB download) | 0.25 / 0.5 / 70 (EN defaults) | re-screen Whisper WER ≤ 10 % (dataset ships `normalized_wer`, threshold 15 %) |

- Ratio starts 1 : 2 (KO : EN). Knob: subsample EN if Korean CER lags at the
  80k/120k gates (see §5).
- Prepare: `TINY_VOCA_FRONTEND=own` on EN shards (6-process parallel, ~20 min);
  KSS features already exist.
- Manifest merge: `scripts/merge_annotations.py` → `annotations_mixed/`
  (train/val/test) declaring both annotation fingerprints + both front-end
  fingerprints; `features_mixed/` with directory junctions (`ko` → `features2`,
  `en` → `features_en`); per-row path prefixes.
- `data.py` patch (~5 lines, `AnnotatedDataset.__init__`): accept rows whose
  `frontend`/`annotation` is in the declared sets from `annotations.json`
  (replaces the single-value check; keeps fail-fast semantics).
- Split discipline: KSS group-wise-by-sentence + EN sentence-grouped splits;
  existing `check_leakage` must pass on the merged manifests.

## 4. Model / training spec (Phase 3)

- `config-v2.json` = `config-ko.json` + `"speakers": ["KSS", "F2", "M2"]`
  (vocab 142, same 24 kHz / hop 256 / 80-mel grid).
- Architecture unchanged (4.5M budget; D5).
- Recipe (proven from TRAIN.md, scaled): `--steps 160000 --batch-frames 12000
  --max-batch 48 --learning-rate 3e-4 --warmup 1000 --lr-half-life 100000
  --gan-start 8000 --gan-ramp 3000 --gan-ratio 0.7 --feature-matching 30
  --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 --validation-size 32
  --save-every 2000 --keep-every 10000 --log-every 50 --max-seconds 14400`.
- Budget: ~14 h GPU (≈ 837 steps/epoch over 26 h mixed corpus; 160k ≈ 191 epochs).
- Smoke first (operating rule): 100 clips / 800 steps, listen, then full run via
  `scripts/train_full.ps1` chunking.

## 5. Evaluation gates

| Check | Gate | Tool |
| --- | --- | --- |
| Korean CER, v1.0 test split | ≤ 12.5 % (v1.0: 11.8 %) | `tiny_voca.evaluate` (ko ASR) |
| English WER, EN test split | ≤ 1.5 % (original model ≈ 0.9 %) | `tiny_voca.evaluate` (small.en) |
| Loanword battery (20 words) | 100 % correct syllables | listening |
| Number battery (20 cases: counters/phones/dates/percent) | 100 % correct reading | listening + ASR |
| Mode A code-mixed (10 sentences) | coherent KSS voice across English spans | listening |
| Mode B regression | voice quality unchanged vs v1-era English baseline | WER + listening |

Decision points: 80k (trends), **120k (D1 fallback gate)**, 160k (final).

## 6. Phase plan

| Phase | Work | Est. |
| --- | --- | --- |
| P0 | `git restore .` of deleted artifacts (runs/, features2/, annotations_full/, …); download EN dataset | 0.5 d |
| P1 | Front end: TN rules + loanword G2P + dictionary seed + `test_frontend_gate.py` ≥ 120 cases, all green | 2–4 d |
| P2 | EN features + alignment + screening; merge manifests; `data.py` patch | 0.5 d + ~1 h GPU |
| P3 | Smoke → full 160k training; checkpoint probes at 20k/80k/120k | ~14 h GPU |
| P4 | Evaluation gates; v2.0 sample battery; `FINAL_REPORT_v2.md` | 0.5 d |

Out of scope for v2.0: browser demo (`web/public`) update — tracked as P5 after gates pass.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Korean regression from from-scratch bilingual (D1) | 120k gate; warm-start branch machinery already built |
| Loanword dictionary coverage | < 2 % fallback gate; dictionary growth loop |
| EN synthetic voices (Qwen) differ from real KSS voice | irrelevant within a mode: Mode A is KSS-tokens-only, Mode B is EN-only |
| EN alignment QC too strict → corpus shrinks | EN defaults are tuned for this aligner/corpus; adjust per summary quantiles |
| License: KSS is CC BY-NC-SA 4.0 | v2.0 weights inherit the non-commercial clause; document in report |

## 8. Open questions

1. ~~Q1 — Loanword dictionary seed~~ — **RESOLVED**: LLM-drafted + human review.
   LLM generates a high-coverage seed (everyday loanwords, tech/business, brands,
   units); edge cases flagged for user review. Rule-based syllabifier must handle
   unknown/OOV English deterministically (no leaks, no crashes) in Mode A.
2. ~~Q2 — Price/currency readings (`$12.99`)~~ — **RESOLVED** (user baseline):
   `$12.99` → `십이 달러 구십구 센트`; `$12` → `십이 달러`; `3/14` date → `삼월 십사일`,
   ratio → `십사 분의 삼` (or `삼 슬래시 십사`); `010-1234-5678` → digit-by-digit
   `공일공 …`; `3개` → `세 개` vs `3층` → `삼 층`. Full table in §2.2 (sign-off pending).
3. Q3 — Step budget 160k vs 190k — decide at the 120k gate based on CER/WER
   trends.
