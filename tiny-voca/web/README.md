# 데모 사이트 — `web/public`

브라우저에서 이중 언어 TTS를 돌리는 정적 페이지입니다. 빌드 단계가 없고, `public/`의 파일 그대로를
어느 정적 호스팅에 올리기만 하면 됩니다.

| 파일 | 내용 |
| --- | --- |
| `index.html` | 이중 언어 페이지. 음성 선택기(KSS·F2·M2), 예문 25개(한국어·숫자·한영 혼용·영어 F2·영어 M2 유형당 5개, 누르면 바로 재생), 빠르기·음높이 슬라이더, 합성 지표, 음소 보기 |
| `worker.js` | 워커. Pyodide에서 `kofront`(g2pk + dizione)로 음소화 — 음성에 따라 모드 A(`ko-loan`, KSS)·모드 B(`own`, F2/M2) 선택 — ONNX Runtime Web으로 추론 |
| `en.html`, `worker-en.js` | 업스트림의 영어 데모 (동일 구조, F2/M2 2음성) |
| `404.html` | Pages/Cloudflare용 안내 페이지 |
| `assets/` | 모델·사전·Python 패키지 (아래 표) |

## 로컬에서 보기

```powershell
cd C:\ai-workspace\korean-tts
.\.venv\Scripts\python.exe -m http.server 8081 --directory tiny-voca\web\public
# http://127.0.0.1:8081/
```

첫 방문 시 약 40 MB(모델 21.4 MB + 발음 사전 + Pyodide 런타임)를 받고,
두 번째 방문부터는 Cache Storage에서 바로 시작합니다.

## 자산 다시 만들기

```powershell
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko-loan"   # v2: 이중언어 체크포인트는 ko-loan 어휘로 내보낸다
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"

.\.venv\Scripts\python.exe scripts\build_mecab_dict.py      # 형태 분석 사전 (MeCab 분석 기록)
.\.venv\Scripts\python.exe scripts\build_web_assets.py --checkpoint runs\v2\inference.pt --export
```

| 자산 | 크기 | 출처 |
| --- | --- | --- |
| `model.onnx`, `config.json` | 21.4 MB | `runs/v2/inference.pt` → `python -m tiny_voca.export` |
| `phonemizer/` | 6.3 MB | `weights/phonemizer` (dizione) + NLTK `cmudict.zip` |
| `mecab-dict.json` | 0.95 MB | 학습 스플릿 12 210문장의 MeCab 분석 (20 649 어절) |
| `pyfront.zip` | 0.04 MB | g2pk + jamo + `kofront`(vocab/text/korean) + `mecab.py` shim |
| `dizione.zip` | 0.06 MB | `dizione/src/dizione` |

## 검증 게이트

```powershell
.\.venv\Scripts\python.exe scripts\check_onnx.py            # ONNX 그래프 == torch 모델
.\.venv\Scripts\python.exe scripts\check_shim.py            # shim == MeCab (보유 스플릿)
```

* ONNX: 지속 시간 완전 일치, 상관 0.982 이상(내부 노이즈 실현값만 다름), `--external-noise`
  그래프는 파형 최대 오차 6e-4.
* shim: 보유 스플릿 462문장에서 토큰 열 **92.6 % 완전 일치**, 토큰 정확도 **99.80 %**.
  차이는 된소리/연음 규칙 하나씩이며, 원인은 `python-mecab-ko`가 WASM에 없는 것.

## 배포

* **GitHub Pages** — `.github/workflows/pages.yml`이 `tiny-voca/web/public`을 그대로 게시합니다.
  → **[ooodev.github.io/korean-tts](https://ooodev.github.io/korean-tts/)**
* **Cloudflare Pages** — `web/wrangler.jsonc`를 사용 (`npx wrangler pages deploy public`).
* **임의의 정적 호스팅** — `public/`을 복사하면 끝. WASM용 응답 헤더는 기본값으로 충분합니다.

## 게시본 실측 (headless Chrome)

`main`에 푸시하면 워크플로우가 게시합니다. 게시 뒤 예문 칩을 눌러 합성까지 돌린 실측값
(상태 줄이 `완료.`가 되고 `#result`의 `<audio>`가 채워지는 것까지 확인):

v2 (2026-11-18, 스텝 160000):

| 예문 | 음성/모드 | 오디오 | 합성 벽시계 | 배 실시간 | 음소 분석 |
| --- | --- | --- | --- | --- | --- |
| 숫자 | KSS / 모드 A | 4.4초 | 0.28초 | 15.7배 | 33 ms |
| 한영 혼용 | KSS / 모드 A | 3.5초 | 0.21초 | 16.2배 | 64 ms |
| 영어 | F2 / 모드 B | 4.1초 | 0.35초 | 11.9배 | 149 ms |

페이지 오류 0건, 첫 방문 로딩 6.5초. `숫자`는 `ᅵ처니시븅년…`으로
렌더링됩니다. 한영 혼용의 `Python`은 차용어 사전 표기 `파ᅵ썬ᅳ로`(파이썬)로 —
`data/loanwords.tsv`를 zip에 함께 배포하기 전에는 OOV 폴백 표기가 나갔던 부분입니다.
영어 칩은 `ɪtwʌzɐbɹaɪt…`로 dizione IPA(모드 B)가 그대로 나갑니다.

v1 (2026-11-16): 숫자 4.7초/0.28초(16.9배), 음운 규칙 4.7초/0.29초(16.2배),
한영 혼용(로컬 서버) 3.6초/0.37초(9.7배, 186 ms 차가운 캐시) — 페이지 오류 0건.

## 알려진 한계

* 세 음성은 각각의 코퍼스로 학습되었습니다: KSS는 한국어(영어 차용어는 모드 A 한국식 발음),
  F2·M2는 영어. 음성을 바꾸면 프론트엔드도 따라 바뀝니다(음성 = 모드).
* 형태 분석 shim은 코퍼스 사전에 없는 어절에서 규칙 하나를 다르게 선택할 수 있습니다.
* 합성은 WASM 단일 스레드에서 실시간의 약 10~17배로 동작합니다 (4초 문장 ≈ 0.3~0.4초).
  첫 문장은 워밍업으로 이보다 느립니다.
