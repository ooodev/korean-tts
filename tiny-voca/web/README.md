# 데모 사이트 — `web/public`

브라우저에서 한국어 TTS를 돌리는 정적 페이지입니다. 빌드 단계가 없고, `public/`의 파일 그대로를
어느 정적 호스팅에 올리기만 하면 됩니다.

| 파일 | 내용 |
| --- | --- |
| `index.html` | 한국어 페이지. 예문 6개, 빠르기·음높이 슬라이더, 합성 지표, 자모 보기 |
| `worker.js` | 워커. Pyodide에서 `kofront`(g2pk + dizione)로 음소화, ONNX Runtime Web으로 추론 |
| `en.html`, `worker-en.js` | 업스트림의 영어 데모 (동일 구조, F2/M2 2음성) |
| `404.html` | Pages/Cloudflare용 안내 페이지 |
| `assets/` | 모델·사전·Python 패키지 (아래 표) |

## 로컬에서 보기

```powershell
cd C:\ai-workspace\korean-tts
.\.venv\Scripts\python.exe -m http.server 8081 --directory tiny-voca\web\public
# http://127.0.0.1:8081/
```

첫 방문 시 약 40 MB(모델 21.4 MB + 발음 사전 + Pyodide 런타임)를 받고, 두 번째 방문부터는
Cache Storage에서 바로 시작합니다.

## 자산 다시 만들기

```powershell
$env:PYTHONUTF8="1"; $env:TINY_VOCA_FRONTEND="ko"
$env:DIZIONE_DATA="C:\ai-workspace\korean-tts\weights\phonemizer"

.\.venv\Scripts\python.exe scripts\build_mecab_dict.py      # 형태 분석 사전 (MeCab 분석 기록)
.\.venv\Scripts\python.exe scripts\build_web_assets.py      # assets/ 전체 조립 (ONNX 포함은 --export)
```

| 자산 | 크기 | 출처 |
| --- | --- | --- |
| `model.onnx`, `config.json` | 21.4 MB | `runs/full/inference.pt` → `python -m tiny_voca.export` |
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

## 게시본 실측 (headless Chrome, 2026-11-16)

`main`에 푸시하면 워크플로우가 게시합니다. 게시 뒤 예문 칩을 눌러 합성까지 돌린 실측값
(상태 줄이 `완료.`가 되고 `#result`의 `<audio>`가 채워지는 것까지 확인):

| 예문 | 오디오 | 합성 벽시계 | 배 실시간 | 음소 분석 |
| --- | --- | --- | --- | --- |
| 숫자 | 4.7초 | 0.28초 | 16.9배 | 29 ms (따뜻한 캐시) |
| 음운 규칙 | 4.7초 | 0.29초 | 16.2배 | – |
| 한영 혼용 (로컬 서버) | 3.6초 | 0.37초 | 9.7배 | 186 ms (차가운 캐시) |

페이지 오류 0건. `숫자`는 `ᅵ처니시븅년…`, `음운 규칙`은
`궁물마시끈내…`(국물→[궁물], 끝내주고→[끈내주고])로 렌더링됩니다 — 음운 규칙이
자모 단계에서 이미 적용된다는 뜻입니다. 한영 혼용은 `ᅩ늘paɪθɑːnᅳ로…`로 dizione 음소가
한국어 자모 사이에 그대로 들어갑니다.

## 알려진 한계

* 영어 음성(68개 음소)은 KSS에 없어 학습되지 않았습니다 — 앞부분은 정확히 읽습니다.
* 형태 분석 shim은 코퍼스 사전에 없는 어절에서 규칙 하나를 다르게 선택할 수 있습니다.
* 합성은 WASM 단일 스레드에서 실시간의 약 10~17배로 동작합니다 (4초 문장 ≈ 0.3~0.4초).
  첫 문장은 워밍업으로 이보다 느립니다.
