# Publish the v2.0 bilingual browser demo.
#
# Run AFTER the final gate passes, from anywhere:
#   powershell -File scripts\publish_v2_demo.ps1
#
# It re-derives the demo assets from the trained checkpoint, bumps the cache
# VERSION, and runs the two demo gates. What it deliberately does NOT do:
# fill the <!-- PUBLISH --> markers in index.html / README / GUIDE / report
# (they need the final gate numbers) and commit/push (GitHub Pages publishes
# tiny-voca/web/public on push).

param(
    [string]$Checkpoint = "runs\v2\inference.pt",
    [int]$ShimSentences = 644
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONUTF8 = "1"
$env:TINY_VOCA_FRONTEND = "ko-loan"          # the v2 checkpoint speaks the 142-token inventory
$env:DIZIONE_DATA = "$root\weights\phonemizer"
$py = "$root\.venv\Scripts\python.exe"

Write-Host "== 1/6  MeCab dictionary (KSS train split)" -ForegroundColor Cyan
& $py scripts\build_mecab_dict.py

Write-Host "== 2/6  Web assets (ONNX re-export under ko-loan)" -ForegroundColor Cyan
& $py scripts\build_web_assets.py --checkpoint $Checkpoint --export

Write-Host "== 3/6  Strict export (--external-noise) for the exact parity check" -ForegroundColor Cyan
& $py -m tiny_voca.export --checkpoint $Checkpoint --output runs\v2\model-noise.onnx --external-noise

Write-Host "== 4/6  Bump VERSION in worker.js (Cache Storage key)" -ForegroundColor Cyan
$workerPath = "tiny-voca\web\public\worker.js"
$text = Get-Content -Raw (Join-Path $root $workerPath)
if ($text -match 'const VERSION = "(\d+)";') {
    $next = [int]$Matches[1] + 1
    $text = $text -replace 'const VERSION = "\d+";', ('const VERSION = "' + $next + '";')
    [System.IO.File]::WriteAllText((Join-Path $root $workerPath), $text, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "    VERSION bumped to $next"
} else {
    throw "const VERSION not found in $workerPath"
}

Write-Host "== 5/6  Gate: ONNX graph == torch model" -ForegroundColor Cyan
& $py scripts\check_onnx.py --checkpoint $Checkpoint `
    --onnx tiny-voca\web\public\assets\model.onnx `
    --strict runs\v2\model-noise.onnx

Write-Host "== 6/6  Gate: shim == MeCab (held-out splits, the shipped dictionary)" -ForegroundColor Cyan
$env:MECAB_DICT = "$root\tiny-voca\web\public\assets\mecab-dict.json"
& $py scripts\check_shim.py --sentences $ShimSentences

Write-Host ""
Write-Host "Assets published and both demo gates green. Remaining manual steps:" -ForegroundColor Yellow
Write-Host "  1. Fill the <!-- PUBLISH --> markers with the final gate numbers:"
Write-Host "       tiny-voca\web\public\index.html   (CER/WER, ONNX size, elapsed time)"
Write-Host "       tiny-voca\web\README.md           (ONNX size, published-site measurement)"
Write-Host "       GUIDE.md                          (section 3 measured-in-Chrome row)"
Write-Host "       FINAL_REPORT_v2.md                (sections 5-8)"
Write-Host "  2. git add/commit/push — the pages workflow publishes tiny-voca/web/public."
