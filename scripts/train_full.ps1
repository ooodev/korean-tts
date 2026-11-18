<#
    Run the full training in wall-clock chunks until --steps is reached.

    Each invocation of tiny_voca.train stops at --chunk-seconds, saves last.pt and
    inference.pt, and the next iteration resumes from that checkpoint, so a dropped
    session never loses more than the chunk in flight. A chunk that dies (OOM, driver
    hiccup) is retried from the same checkpoint.

    powershell -File scripts/train_full.ps1                     # defaults: runs/full, 104k steps
    powershell -File scripts/train_full.ps1 -Steps 60000 -ChunkSeconds 10800
#>
param(
    [int]$Steps = 104000,
    [int]$ChunkSeconds = 14400,
    [string]$Output = "runs/full",
    [string]$Annotations = "annotations_full",
    [string]$Config = "config-ko.json",
    [string]$Log = "logs_train_full.txt",
    [int]$MaxAttempts = 40,
    [string]$Frontend = "ko",   # v2.0 trains with ko-loan
    [string]$InitFrom = "",     # D1 branch: start from another run's checkpoint
    [switch]$NewData            # D1 branch: --init-from data comes from different annotations
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONWARNINGS = "ignore"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:DIZIONE_DATA = "C:\ai-workspace\korean-tts\weights\phonemizer"
$env:TINY_VOCA_FRONTEND = $Frontend
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$python = ".venv\Scripts\python.exe"
$last = Join-Path $Output "last.pt"

function Current-Step {
    if (-not (Test-Path $last)) { return 0 }
    # torch can print warnings to stdout, so take the last line and force a scalar:
    # an array on the left of -ge makes the comparison return an array, which is
    # always truthy, and the loop would stop early.
    $line = (& $python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" $last |
             Select-Object -Last 1)
    if (-not $line) { return 0 }
    return [int]$line
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $step = Current-Step
    if ($step -ge $Steps) { break }
    "=== attempt $attempt starting from step $step (target $Steps) ===" | Out-File -FilePath $Log -Append -Encoding utf8

    $arguments = @(
        "-m", "tiny_voca.train",
        "--train", (Join-Path $Annotations "train.jsonl"),
        "--validation", (Join-Path $Annotations "validation.jsonl"),
        "--output", $Output, "--config", $Config, "--device", "cuda",
        "--steps", "$Steps",
        "--batch-frames", "12000", "--max-batch", "48",
        "--learning-rate", "3e-4", "--warmup", "1000", "--lr-half-life", "100000",
        "--gan-start", "8000", "--gan-ramp", "3000", "--gan-ratio", "0.7",
        "--teacher-hold", "3000", "--teacher-ramp", "7000", "--teacher-final", "0",
        "--validation-size", "32",
        "--save-every", "2000", "--keep-every", "10000", "--log-every", "50",
        "--max-seconds", "$ChunkSeconds"
    )
    if (Test-Path $last) {
        $arguments += @("--resume", $last)
    } elseif ($InitFrom) {
        # --init-from and --resume are mutually exclusive in train.py, so the
        # branch flags go on the first chunk only; later chunks resume as usual.
        $arguments += @("--init-from", $InitFrom)
        if ($NewData) { $arguments += @("--new-data") }
    }

    & $python @arguments *>> $Log
    if ($LASTEXITCODE -ne 0) { "!!! chunk exited with $LASTEXITCODE" | Out-File -FilePath $Log -Append -Encoding utf8 }
}

"=== training finished at step $(Current-Step) ===" | Out-File -FilePath $Log -Append -Encoding utf8
