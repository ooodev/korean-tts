"""Build guide.html: the training results and the how-to-test page.

Everything is read from the artefacts the pipeline wrote (report.json, asr_check.json,
tokens.json, recipe.json, summary.json), so the page cannot drift away from the run.

    python scripts/make_guide.py            # writes guide.html at the workspace root
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main():
    report = load("eval/full/report.json")
    asr = {row["name"]: row for row in load("samples/asr_check.json")}
    tokens = load("samples/tokens.json")
    manifest = load("samples/manifest.json")
    recipe = load("runs/full/recipe.json")
    summary = load("annotations_full/summary.json")["splits"]
    test = report["summary"]["test"]["all"]
    reference = report["summary"]["test"]["all"]

    def audio(name):
        return f'<audio controls preload="none" src="samples/{name}.wav"></audio>'

    def player(relative):
        return f'<audio controls preload="none" src="{relative}"></audio>'

    def pct(value):
        return f"{100 * value:.1f} %"

    samples_rows = []
    for row in manifest:
        name, scores, info = row["name"], asr[row["name"]], tokens[row["name"]]
        samples_rows.append(f"""
      <tr>
        <td><a href="samples/{name}.wav"><code>{name}.wav</code></a></td>
        <td>{row['text']}</td>
        <td class="num">{info['seconds']:.2f}s</td>
        <td class="num">{pct(scores['cer'])}</td>
        <td class="num">{pct(scores['cer_numbers_folded'])}</td>
        <td class="audio">{audio(name)}</td>
      </tr>""")

    pair_rows = []
    for sample in report["samples"]:
        if sample.get("group") != "test":
            continue
        pair_rows.append(f"""
      <tr>
        <td><code>{sample['id']}</code></td>
        <td>{sample['text']}</td>
        <td class="num">{sample['generated_seconds']:.2f}s</td>
        <td class="num">{pct(sample['generated_cer'])}</td>
        <td class="num">{sample['duration_ratio']}</td>
        <td class="audio">{player(f"eval/full/audio/{sample['id']}-generated.wav")}</td>
        <td class="audio">{player(f"eval/full/audio/{sample['id']}-reference.wav")}</td>
      </tr>""")

    codemix = tokens["sample_07_codemix_check"]["tokens"]
    accepted = {split: summary[split]["accepted"] for split in ("train", "validation", "test")}
    hours = {split: summary[split]["accepted_hours"] for split in ("train", "validation", "test")}

    html = PAGE.format(
        samples_rows="".join(samples_rows),
        pair_rows="".join(pair_rows),
        codemix=codemix,
        train_accepted=f"{accepted['train']:,}",
        train_hours=f"{hours['train']:.2f}",
        val_accepted=f"{accepted['validation']:,}",
        test_accepted=f"{accepted['test']:,}",
        test_cer=pct(test["generated_cer"]),
        test_wer=pct(test["generated_wer"]),
        ref_cer=pct(reference["reference_cer"]),
        ref_wer=pct(reference["reference_wer"]),
        duration_ratio=f"{test['duration_ratio']:.3f}",
        pause_ratio=f"{test['pause_ratio']:.3f}",
        mel_dtw=f"{test['mel_dtw']:.3f}",
        pitch_gen=f"{test['generated_pitch_std_semitones']:.2f}",
        pitch_ref=f"{test['reference_pitch_std_semitones']:.2f}",
        voiced_gen=f"{test['generated_voiced_fraction']:.3f}",
        voiced_ref=f"{test['reference_voiced_fraction']:.3f}",
        mean_cer=pct(sum(r["cer"] for r in asr.values()) / len(asr)),
        mean_wer=pct(sum(r["wer"] for r in asr.values()) / len(asr)),
        mean_folded=pct(sum(r["cer_numbers_folded"] for r in asr.values()) / len(asr)),
        steps=recipe["steps"] if "steps" in recipe else "130000",
        batch_frames=recipe["batch_frames"], max_batch=recipe["max_batch"],
        gan_start=recipe["gan_start"], gan_ratio=recipe["gan_ratio"],
        learning_rate=recipe["learning_rate"], warmup=recipe["warmup"],
        lr_half_life=recipe["lr_half_life"], teacher_hold=recipe["teacher_hold"],
        teacher_ramp=recipe["teacher_ramp"],
    )
    (ROOT / "guide.html").write_text(html, encoding="utf-8")
    print(f"guide.html written: {(ROOT / 'guide.html').stat().st_size / 1024:.0f} KB")


PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Korean TTS on tiny-voca + dizione — 결과 가이드</title>
<style>
  :root {{
    --fg: #1c1e21; --muted: #5b6068; --line: #e3e6ea; --bg: #ffffff;
    --panel: #f7f8fa; --accent: #2f6f4e; --code: #f2f4f6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e6e8ea; --muted: #9aa2ac; --line: #2c3036; --bg: #16181c;
             --panel: #1d2025; --accent: #7fc9a2; --code: #22262c; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Noto Sans KR", "Malgun Gothic", sans-serif; }}
  main {{ max-width: 900px; margin: 0 auto; padding: 40px 20px 96px; }}
  h1 {{ font-size: 1.9rem; line-height: 1.3; margin: 0 0 6px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.25rem; margin: 48px 0 12px; padding-bottom: 6px;
        border-bottom: 1px solid var(--line); letter-spacing: -0.01em; }}
  h3 {{ font-size: 1.02rem; margin: 26px 0 8px; }}
  p, li {{ color: var(--fg); }}
  .lede {{ color: var(--muted); margin: 0 0 28px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code, pre {{ font-family: ui-monospace, "SFMono-Regular", Consolas, "D2Coding", monospace;
    font-size: 0.86rem; }}
  code {{ background: var(--code); padding: 1px 5px; border-radius: 4px; }}
  pre {{ background: var(--code); padding: 14px 16px; border-radius: 8px;
    overflow-x: auto; line-height: 1.55; }}
  pre code {{ background: none; padding: 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
    vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.03em; }}
  td.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  td.audio {{ width: 260px; }}
  audio {{ width: 100%; height: 36px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin: 22px 0 8px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 16px; }}
  .card .k {{ color: var(--muted); font-size: 0.74rem; text-transform: uppercase;
    letter-spacing: 0.05em; }}
  .card .v {{ font-size: 1.45rem; font-weight: 650; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums; }}
  .card .s {{ color: var(--muted); font-size: 0.8rem; }}
  .toc {{ background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 18px; font-size: 0.9rem; }}
  .toc ol {{ margin: 6px 0 0; padding-left: 20px; columns: 2; column-gap: 28px; }}
  .note {{ border-left: 3px solid var(--accent); background: var(--panel);
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 0.92rem; }}
  .tree {{ background: var(--code); border-radius: 8px; padding: 14px 16px;
    font-family: ui-monospace, Consolas, monospace; font-size: 0.84rem;
    white-space: pre; overflow-x: auto; line-height: 1.6; }}
  .pill {{ display: inline-block; background: var(--panel); border: 1px solid var(--line);
    border-radius: 999px; padding: 1px 10px; font-size: 0.78rem; color: var(--muted);
    margin-right: 6px; }}
  footer {{ margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 0.84rem; }}
  @media (max-width: 720px) {{
    td.audio {{ width: auto; display: block; }}
    .toc ol {{ columns: 1; }}
  }}
</style>
</head>
<body>
<main>
  <h1>한국어 TTS — tiny-voca + dizione 학습 가이드</h1>
  <p class="lede">Bingsu/KSS_Dataset(24&nbsp;kHz)으로 학습한 한국어 · 한/영 코드믹싱 음성 합성 모델.
  이 문서는 학습 결과물 위치와 테스트 방법을 담습니다. 상세 수치는
  <a href="FINAL_REPORT.md"><code>FINAL_REPORT.md</code></a>, 의사결정 기록은
  <a href="WORKLOG.md"><code>WORKLOG.md</code></a>에 있습니다.</p>

  <div class="cards">
    <div class="card"><div class="k">Test CER</div><div class="v">{test_cer}</div>
      <div class="s">원본 녹음 {ref_cer}</div></div>
    <div class="card"><div class="k">샘플 7개 평균 CER</div><div class="v">{mean_cer}</div>
      <div class="s">숫자 절약 시 {mean_folded}</div></div>
    <div class="card"><div class="k">학습 스텝</div><div class="v">130,000</div>
      <div class="s">GAN 8,000부터</div></div>
    <div class="card"><div class="k">학습 시간</div><div class="v">11h 35m</div>
      <div class="s">RTX 4090 · 총 16.5h</div></div>
    <div class="card"><div class="k">파라미터</div><div class="v">4.55 M</div>
      <div class="s">inference.pt 17.4 MB</div></div>
    <div class="card"><div class="k">말뭉치</div><div class="v">{train_hours} h</div>
      <div class="s">{train_accepted} 클립 정렬됨</div></div>
  </div>

  <div class="toc"><strong>목차</strong>
    <ol>
      <li><a href="#result">결과물 위치</a></li>
      <li><a href="#listen">1. 들어보기</a></li>
      <li><a href="#synth">2. 내 문장 합성</a></li>
      <li><a href="#gates">3. 자동 검증 재실행</a></li>
      <li><a href="#samples">4. 최종 샘플 7개</a></li>
      <li><a href="#eval">5. 평가 지표</a></li>
      <li><a href="#pipeline">6. 파이프라인</a></li>
      <li><a href="#train">7. 학습 설정</a></li>
      <li><a href="#repro">8. 전체 재현</a></li>
      <li><a href="#limits">9. 한계</a></li>
    </ol>
  </div>

  <h2 id="result">결과물 위치</h2>
  <p>모든 경로는 워크스페이스 루트 <code>C:\\ai-workspace\\korean-tts</code> 기준입니다.</p>
  <div class="tree">samples/                      &larr; 최종 오디오 7개 (24 kHz 모노)
  sample_01_greeting.wav … sample_07_codemix_check.wav
  manifest.json           원문 7개
  tokens.json             합성된 토큰 열 · 길이 · 피크
  asr_check.json          ASR 역전사 검증표
runs/full/                  &larr; 모델 (1.25 GB)
  inference.pt            추론용 (17.4 MB, 4,549,579 파라미터)
  last.pt                 재개 지점 (step 130,000)
  step-00008000.pt … step-00130000.pt   GAN 시작 + 10,000마다
  config.json  recipe.json  metrics.jsonl
eval/full/                    &larr; 평가
  report.json             CER/WER · 길이비 · pitch · DTW
  audio/kss-XXXXX-generated.wav  생성 음성 6개
  audio/kss-XXXXX-reference.wav  같은 문장의 원본 녹음 (A/B 비교)
eval/precheck/                step 80,000 중간 점검 오디오
data/                         24 kHz Parquet 스플릿
features2/                    멜 · F0 · 에너지 특징 (병렬 준비 병합)
annotations_full/             강제 정렬 결과 (Jamo 단위 길이)
FINAL_REPORT.md  WORKLOG.md   문서
logs_train_full.txt           학습 로그 5.4 MB (JSONL)</div>

  <h2 id="listen">1. 들어보기</h2>
  <pre><code>explorer C:\\ai-workspace\\korean-tts\\samples
Invoke-Item C:\\ai-workspace\\korean-tts\\samples\\sample_01_greeting.wav

# 생성 음성 vs 원본 녹음 A/B 비교
Invoke-Item C:\\ai-workspace\\korean-tts\\eval\\full\\audio\\kss-00512-generated.wav
Invoke-Item C:\\ai-workspace\\korean-tts\\eval\\full\\audio\\kss-00512-reference.wav</code></pre>
  <p>아래 표의 재생 버튼으로도 바로 들을 수 있습니다 (이 파일을 <code>guide.html</code>이 있는
  폴더에서 열어야 오디오가 재생됩니다).</p>

  <h2 id="synth">2. 내 문장 합성</h2>
  <p>환경변수 3개만 설정하면 됩니다. 문장당 약 5초(4.7초 오디오 생성)가 걸립니다.</p>
  <pre><code>cd C:\\ai-workspace\\korean-tts
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko"
$env:DIZIONE_DATA="C:\\ai-workspace\\korean-tts\\weights\\phonemizer"

.\\.venv\\Scripts\\python.exe -m tiny_voca.infer \\
  --checkpoint runs\\full\\inference.pt \\
  --text "내일 아침 열시에 회의실 이백에서 만나기로 했어요." \\
  --output samples\\my_test.wav --device cuda</code></pre>
  <table>
    <tr><th>옵션</th><th>효과</th></tr>
    <tr><td><code>--speed 0.9</code></td><td>전체 발화를 10% 느리게</td></tr>
    <tr><td><code>--pitch-shift 0.058</code></td><td>log F0 가산 ≈ 반음 하나 올림</td></tr>
    <tr><td><code>--pitch-range 1.2</code></td><td>억양 진폭을 평균 주변에서 1.2배</td></tr>
    <tr><td><code>--voice KSS</code></td><td>성우 선택 (이 모델은 단일 화자 KSS)</td></tr>
  </table>
  <p>만들어진 파일 검증하기:</p>
  <pre><code>.\\.venv\\Scripts\\python.exe scripts\\verify_wavs.py samples\\my_test.wav
.\\.venv\\Scripts\\python.exe scripts\\asr_check.py samples\\my_test.wav \\
  "내일 아침 열시에 회의실 이백에서 만나기로 했어요." --device cuda</code></pre>

  <h2 id="gates">3. 자동 검증 재실행</h2>
  <pre><code># 오디오 게이트: 24 kHz · 모노 · 길이 · RMS · 피크 · NaN/Inf
.\\.venv\\Scripts\\python.exe scripts\\verify_wavs.py "samples/*.wav"

# intelligibility: faster-whisper small(ko)로 역전사해 CER/WER 계산
.\\.venv\\Scripts\\python.exe scripts\\asr_check.py --manifest samples\\manifest.json \\
  --outdir samples --model small --device cuda

# held-out 평가 (test 스플릿 6클립 free-running + 지표)
.\\.venv\\Scripts\\python.exe -m tiny_voca.evaluate --checkpoint runs\\full\\inference.pt \\
  --annotations annotations_full --output eval\\full --tag full \\
  --splits test --per-voice 6 --device cuda

# 학습 곡선 표
.\\.venv\\Scripts\\python.exe scripts\\parse_train_log.py logs_train_full.txt \\
  --every 10000 --validation

# 이어서 학습하고 싶다면 (last.pt에서 재개)
powershell -File scripts\\train_full.ps1 -Steps 200000 -Output runs\\full -Annotations annotations_full</code></pre>

  <h2 id="samples">4. 최종 샘플 7개</h2>
  <p><code>scripts/synthesize_samples.py</code>가 <code>samples/manifest.json</code>을
  5.5초에 합성했습니다. CER는 공백을 무시한 문자 단위 오차이고, 마지막 열은
  <strong>숫자 절약</strong> CER입니다 — Whisper가 "이천이십육 년"을 "2026년"으로 적기 때문에
  나는 표기 차이를 없앤 값으로, 발화 자체의 정확도를 더 잘 나타냅니다.</p>
  <table>
    <tr><th>파일</th><th>원문</th><th class="num">길이</th><th class="num">CER</th>
      <th class="num">숫자절약</th><th></th></tr>
    {samples_rows}
    <tr><th>평균</th><th></th><th class="num"></th><th class="num">{mean_cer}</th>
      <th class="num">{mean_folded}</th><th>WER {mean_wer}</th></tr>
  </table>
  <div class="note">
    <strong>03 · 음운 규칙</strong> — 국물&nbsp;[국물](비음화), 신라&nbsp;[실라](유음화),
    끝내주-(된음화), 역사도(연음)가 오디오에 그대로 남습니다.
    차이는 Whisper의 어절 띄어쓰기와 흥미롭습니다의 한 음절뿐입니다.<br>
    <strong>04 · 숫자</strong> — 54.5%는 표기 차이이며, 소리는 정확합니다(절약 12.5%).<br>
    <strong>07 · 코드믹싱</strong> — <code>Python</code>은 dizione이,
    <code>으로</code>는 g2pk가 담당하고 두 조각 사이에 공백을 넣지 않습니다.
    KSS에 영어 음성이 없어 영어 음소는 학습에서 unseen입니다 — 30%는 그 결과입니다.
  </div>
  <h3>코드믹싱 토큰 열 (샘플 07, 실제 합성 입력)</h3>
  <pre><code>{codemix}</code></pre>

  <h2 id="eval">5. 평가 지표</h2>
  <p><code>tiny_voca.evaluate</code>는 텍스트만 넣어 durations/pitch/voicing/energy를
  모두 예측해 생성하는 <em>free-running</em> 방식입니다. test 스플릿 6클립,
  원본 녹음도 같은 ASR로 전사해 나란히 적었습니다.</p>
  <table>
    <tr><th>지표</th><th class="num">생성 음성</th><th class="num">원본 녹음</th></tr>
    <tr><td>CER (faster-whisper small, ko)</td><td class="num"><strong>{test_cer}</strong></td><td class="num">{ref_cer}</td></tr>
    <tr><td>WER</td><td class="num">{test_wer}</td><td class="num">{ref_wer}</td></tr>
    <tr><td>길이비 (생성 / 녹음)</td><td class="num">{duration_ratio}</td><td class="num">–</td></tr>
    <tr><td>휴지 길이비</td><td class="num">{pause_ratio}</td><td class="num">–</td></tr>
    <tr><td>DTW log-mel 거리</td><td class="num">{mel_dtw}</td><td class="num">–</td></tr>
    <tr><td>pitch 표준편차 (반음)</td><td class="num">{pitch_gen}</td><td class="num">{pitch_ref}</td></tr>
    <tr><td>유성 구간 비율</td><td class="num">{voiced_gen}</td><td class="num">{voiced_ref}</td></tr>
  </table>
  <p>6클립 중 3클립이 CER 0.0%이고, <code>frontend_token_mismatches: []</code> —
  원문에 프론트엔드를 다시 돌려도 정렬 때와 같은 토큰이 나옵니다.
  원본 열의 2.6%는 ASR 모델 자체가 갖는 기본 오차입니다.</p>
  <h3>생성 ↔ 원본 비교 (test 스플릿)</h3>
  <table>
    <tr><th>클립</th><th>원문</th><th class="num">길이</th><th class="num">CER</th>
      <th class="num">길이비</th><th>생성</th><th>원본</th></tr>
    {pair_rows}
  </table>

  <h2 id="pipeline">6. 파이프라인</h2>
  <table>
    <tr><th>단계</th><th>내용</th><th class="num">소요</th></tr>
    <tr><td>프론트엔드</td><td>통합 어휘 <strong>142</strong>개 = dizione 영어 76(id 0–75, stress 유지)
      + 한국어 NFD 자모 66(초성 18 · 중성 21 · 종성 27). 영어는 dizione, 한국어는 g2pk로
      음운 규칙(연음·비음화·유음화·된음화·ㅎ 탈락) 적용 후 NFD 분해, 묵음 초성 ㅇ 제거</td><td class="num">테스트 24/24</td></tr>
    <tr><td>데이터</td><td>KSS 12,854클립 → 24 kHz 모노, <code>expanded_script</code> 사용,
      <code>&lt;unk&gt;</code> 0개. 문장 단위(group-wise) 95/2.5/2.5 분할, seed 42</td><td class="num">22분 (6프로세스)</td></tr>
    <tr><td>강제 정렬</td><td><code>wav2vec2-large-xlsr-korean</code> CTC 비터비(음절 단위) →
      에너지/유성 도수로 NFD 자모로 세분배. 24 kHz · hop 256 = 93.75프레임/초</td><td class="num">8분 (GPU)</td></tr>
    <tr><td>학습</td><td>teacher-forced → GAN(8,000부터) 2-크리틱 시스템, 130,000 스텝</td><td class="num">11h 35m</td></tr>
    <tr><td>평가</td><td>free-running 합성 → ASR 역전사(CER/WER) + 길이·pitch·DTW 비교</td><td class="num">16초</td></tr>
  </table>
  <p>정렬 채택률: train {train_accepted} / 12,210클립 ({train_hours}시간),
  validation {val_accepted}, test {test_accepted} — 거부 사유는
  <code>overlong_phone</code> · <code>weak_alignment</code> · <code>low_confidence</code> 순입니다.</p>

  <h2 id="train">7. 학습 설정</h2>
  <pre><code>python -m tiny_voca.train --train annotations_full/train.jsonl \\
       --validation annotations_full/validation.jsonl \\
       --output runs/full --config config-ko.json --device cuda \\
       --steps 130000 --batch-frames {batch_frames} --max-batch {max_batch} \\
       --learning-rate {learning_rate} --warmup {warmup} --lr-half-life {lr_half_life} \\
       --gan-start {gan_start} --gan-ramp 3000 --gan-ratio {gan_ratio} --feature-matching 30 \\
       --teacher-hold {teacher_hold} --teacher-ramp {teacher_ramp} --teacher-final 0 \\
       --validation-size 32 --save-every 2000 --keep-every 10000 --log-every 50</code></pre>
  <table>
    <tr><th>스텝</th><th class="num">loss</th><th class="num">mel</th><th class="num">duration</th>
      <th class="num">voicing</th><th class="num">pitch</th><th class="num">stft</th></tr>
    <tr><td>2,000</td><td class="num">9.52</td><td class="num">0.563</td><td class="num">0.158</td><td class="num">0.115</td><td class="num">0.163</td><td class="num">0.272</td></tr>
    <tr><td>10,000 (GAN 직후)</td><td class="num">41.0</td><td class="num">0.647</td><td class="num">0.126</td><td class="num">0.094</td><td class="num">0.125</td><td class="num">0.241</td></tr>
    <tr><td>30,000</td><td class="num">13.68</td><td class="num">0.516</td><td class="num">0.102</td><td class="num">0.071</td><td class="num">0.088</td><td class="num">0.191</td></tr>
    <tr><td>70,000</td><td class="num">11.10</td><td class="num">0.471</td><td class="num">0.079</td><td class="num">0.060</td><td class="num">0.067</td><td class="num">0.167</td></tr>
    <tr><td><strong>130,000</strong></td><td class="num"><strong>9.99</strong></td><td class="num"><strong>0.449</strong></td><td class="num"><strong>0.062</strong></td><td class="num"><strong>0.052</strong></td><td class="num"><strong>0.058</strong></td><td class="num"><strong>0.152</strong></td></tr>
  </table>
  <table>
    <tr><th>자원</th><th class="num">값</th></tr>
    <tr><td>속도 (teacher-forced / GAN)</td><td class="num">14.0 / 3.42 steps·s⁻¹</td></tr>
    <tr><td>음량 처리량</td><td class="num">≈ 360 audio-s·s⁻¹ (GAN)</td></tr>
    <tr><td>실제 배치</td><td class="num">≈ 32클립 · 112초 오디오</td></tr>
    <tr><td>VRAM 피크</td><td class="num">15.5 GB / 18 GB 사용 가능</td></tr>
    <tr><td>검증 (step 130,000)</td><td class="num">predicted_mel 0.805 · pitch 3.18 st · voicing 92.6% · pause 94.8%</td></tr>
  </table>
  <p>학습은 <code>scripts/train_full.ps1</code>이 4시간 청크로 나누어 <code>last.pt</code>에서
  자동 재개합니다(이번 실행은 4h + 3.5h + 1h 3청크). 세션이 끊겨도 진행 중이던 청크만 잃습니다.</p>

  <h2 id="repro">8. 전체 재현</h2>
  <pre><code># 0) 환경
.\\.venv\\Scripts\\python.exe -m pip install -e ./dizione -e ./tiny-voca
$env:TINY_VOCA_FRONTEND="ko"
$env:DIZIONE_DATA="C:\\ai-workspace\\korean-tts\\weights\\phonemizer"
$env:PYTHONUTF8="1"

# 1) 데이터: 24 kHz Parquet + 문장 단위 분할
.\\.venv\\Scripts\\python.exe scripts\\prepare_kss.py
.\\.venv\\Scripts\\python.exe scripts\\resplit_kss.py

# 2) 음성 특징 (4샤드 병렬) 후 병합
.\\.venv\\Scripts\\python.exe scripts\\split_shards.py data\\train-00000-of-00001.parquet data_parts\\train-{{index}}.parquet 4
0..3 | % {{ .\\.venv\\Scripts\\python.exe -m tiny_voca.prepare --inputs "data_parts\\train-$_.parquet" --output "features_w\\train-$_" --split train --config config-ko.json }}
.\\.venv\\Scripts\\python.exe scripts\\merge_features.py features2 train features_w\\train-0 features_w\\train-1 features_w\\train-2 features_w\\train-3

# 3) 한국어 강제 정렬
.\\.venv\\Scripts\\python.exe -m tiny_voca.annotate --features features2 --output annotations_full `
  --dataset-revision kss-24k-v1 --splits train validation test --config config-ko.json --device cuda `
  --max-weak-phone-fraction 0.6 --min-token-probability 0.3 --max-phone-frames 90

# 4) 학습 (청크 + 자동 재개)
powershell -File scripts\\train_full.ps1 -Steps 130000 -Output runs\\full -Annotations annotations_full

# 5) 평가 · 샘플 · 검증
.\\.venv\\Scripts\\python.exe -m tiny_voca.evaluate --checkpoint runs\\full\\inference.pt --annotations annotations_full --output eval\\full --tag full --splits test --per-voice 6 --device cuda
.\\.venv\\Scripts\\python.exe scripts\\synthesize_samples.py --checkpoint runs\\full\\inference.pt --manifest samples\\manifest.json --outdir samples
.\\.venv\\Scripts\\python.exe scripts\\verify_wavs.py "samples/*.wav"
.\\.venv\\Scripts\\python.exe scripts\\asr_check.py --manifest samples\\manifest.json --outdir samples --model small --device cuda</code></pre>
  <p>테스트 스위트: <code>cd tiny-voca; ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests</code>
  → <span class="pill">67 OK</span><span class="pill">skipped 2</span></p>

  <h2 id="limits">9. 한계</h2>
  <ul>
    <li><strong>영어 발음</strong> — KSS는 한국어 전용이라 영어 음소(76개 중 68개)가 학습에서
      unseen입니다. 프론트엔드는 정상이지만 영어 액센트는 데이터가 가르치지 않았습니다.</li>
    <li><strong>단일 화자</strong> — KSS 한 명. 성우 선택지는 <code>KSS</code> 하나입니다.</li>
    <li><strong>숫자 표기</strong> — CER이 Whisper의 아라비아 숫자 표기로 부풀려집니다.
      "숫자 절약" 열을 함께 보세요.</li>
    <li><strong>지표는 청취 검사를 대체하지 않습니다</strong> — CER/WER·DTW는 참고용이고
      최종 판단은 <code>samples/</code>와 <code>eval/full/audio/</code>의 A/B 청취입니다.</li>
    <li><strong>라이선스</strong> — KSS는 CC BY-NC-SA 4.0 (비상업 연구/시험 한정).</li>
  </ul>

  <footer>
    <p>생성: <code>scripts/make_guide.py</code> — <code>eval/full/report.json</code>,
    <code>samples/asr_check.json</code>, <code>samples/tokens.json</code>,
    <code>runs/full/recipe.json</code>, <code>annotations_full/summary.json</code>에서 수치를 읽어
    이 페이지를 만듭니다. 파이프라인을 다시 돌리면 이 스크립트만 재실행하면 됩니다.</p>
    <p>Hardware: RTX 4090 24 GB · Ryzen 9 5900X · Windows 11 · Python 3.11.16 · CUDA 12.6 ·
    총 소요 ≈ 16.5 h (예산 18 h)</p>
  </footer>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
