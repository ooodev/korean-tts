"""V2 bilingual gates (V2_SPEC section 5): KO CER on KSS test rows (Mode A) and
EN WER on the English test split (Mode B) for one inference checkpoint.

The KO gate serves the ko-loan front end on the KSS-filtered merged manifest
(single-run annotations_full would fail data.py's strict check under ko-loan);
the EN gate serves the own front end on annotations_en with the English ASR.
Both checkpoints are loaded with check_frontend=False (bilingual checkpoint
stores one front end; the gates deliberately serve the other).

    powershell -File scripts/run_v2_gates.ps1 -Checkpoint runs/v2/inference.pt -Tag gate-80000

Thresholds: KO generated_cer <= 0.125, EN generated_wer <= 0.015.
"""
param(
    [string]$Checkpoint = "runs\v2\inference.pt",
    [string]$Tag = "v2",
    [int]$PerVoiceKo = 300,      # >= 228 KSS test rows
    [int]$PerVoiceEn = 600,      # >= 576 EN test rows
    [double]$KoCerMax = 0.125,
    [double]$EnWerMax = 0.015,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Continue"
Set-Location C:\ai-workspace\korean-tts
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONWARNINGS = "ignore"
$env:DIZIONE_DATA = "C:\ai-workspace\korean-tts\weights\phonemizer"

$python = ".venv\Scripts\python.exe"
$outRoot = "eval\v2\$Tag"
$koManifest = "eval\v2\_ko_gate"
$enManifest = "eval\v2\_en_gate"

# 1. Gate manifests: KSS rows for the KO gate, F2/M2 rows for the EN gate, once
#    (shared by every gate run). Filtered from the merged manifest because the
#    single-run manifests fail data.py's strict check when served under the
#    other front end.
if (-not (Test-Path "$koManifest\test.jsonl")) {
    & $python -X utf8 scripts\gate_manifest.py --annotations annotations_mixed `
        --out $koManifest --speaker KSS --splits test
    if ($LASTEXITCODE -ne 0) { Write-Output "KO_GATE_MANIFEST_FAILED"; exit 1 }
}
if (-not (Test-Path "$enManifest\test.jsonl")) {
    & $python -X utf8 scripts\gate_manifest.py --annotations annotations_mixed `
        --out $enManifest --speaker F2 --speaker M2 --splits test
    if ($LASTEXITCODE -ne 0) { Write-Output "EN_GATE_MANIFEST_FAILED"; exit 1 }
}

# 2. KO gate (Mode A: ko-loan, Korean ASR).
$env:TINY_VOCA_FRONTEND = "ko-loan"
& $python -m tiny_voca.evaluate --checkpoint $Checkpoint `
    --annotations $koManifest --splits test `
    --per-voice $PerVoiceKo --novel 0 --device $Device `
    --output "$outRoot\ko" --tag "$Tag-ko" 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { Write-Output "KO_GATE_FAILED"; exit 1 }

# 3. EN gate (Mode B). The process runs under ko-loan so the 142-token
#    checkpoint loads; --serve-frontend switches phonemisation to the 76-token
#    own inventory for synthesis (its ids are the prefix of the unified one).
$env:TINY_VOCA_FRONTEND = "ko-loan"
& $python -m tiny_voca.evaluate --checkpoint $Checkpoint `
    --annotations $enManifest --splits test --serve-frontend own `
    --per-voice $PerVoiceEn --novel 0 --asr-model small.en --asr-language en `
    --device $Device `
    --output "$outRoot\en" --tag "$Tag-en" 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { Write-Output "EN_GATE_FAILED"; exit 1 }

# 4. Verdict.
& $python -X utf8 -c "
import json, sys
ko = json.load(open(r'$outRoot\ko\report.json', encoding='utf-8'))
en = json.load(open(r'$outRoot\en\report.json', encoding='utf-8'))
ko_cer = ko['summary']['test']['KSS']['generated_cer']
en_wer = en['summary']['test']['all']['generated_wer']
ko_pass = ko_cer is not None and ko_cer <= float(sys.argv[1])
en_pass = en_wer is not None and en_wer <= float(sys.argv[2])
verdict = {
    'tag': '$Tag', 'step': ko.get('step'),
    'ko_cer': ko_cer, 'ko_cer_max': float(sys.argv[1]), 'ko_pass': ko_pass,
    'en_wer': en_wer, 'en_wer_max': float(sys.argv[2]), 'en_pass': en_pass,
    'ko_samples': ko['summary']['test']['KSS']['samples'],
    'en_samples': en['summary']['test']['all']['samples'],
}
verdict['all_pass'] = ko_pass and en_pass
print(json.dumps(verdict, indent=1))
open(r'$outRoot\verdict.json', 'w', encoding='utf-8').write(json.dumps(verdict, indent=1) + '\n')
" $KoCerMax $EnWerMax
exit 0
