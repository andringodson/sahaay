<#
.SYNOPSIS
    Set up Sahaay on a Windows PC, Snapdragon or x86.

.DESCRIPTION
    Creates a virtual environment, installs the right onnxruntime for this
    machine, and optionally downloads model weights.

    The onnxruntime choice is the part that matters. On Windows we install
    onnxruntime-qnn, which carries Qualcomm's plugin execution provider.
    Note that pip-installing it is NOT sufficient on its own - the plugin has
    to be registered at runtime, which sahaay/runtime.py does on import. See
    docs/HARDWARE.md.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Models         # also download weights (~3.7 GB)
    .\install.ps1 -Dev            # include test and AI Hub tooling
#>

[CmdletBinding()]
param(
    [switch]$Models,
    [switch]$Dev,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Say($text, $colour = "Cyan") { Write-Host "  $text" -ForegroundColor $colour }

Write-Host ""
Say "Sahaay setup" "White"
Say ("-" * 46) "DarkGray"

# -- 1. find a suitable Python ------------------------------------------------
# onnxruntime-qnn publishes wheels for Python 3.11+ only. On 3.10 or older we
# fall back to stock onnxruntime, which works but has no NPU path at all.

if ($Python -eq "") {
    $candidates = @()
    foreach ($v in @("3.13", "3.12", "3.11", "3.14")) {
        $found = & py -$v -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $found) { $candidates += $found }
    }
    if ($candidates.Count -eq 0) {
        $found = & python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $found }
    }
    if ($candidates.Count -eq 0) {
        Say "No Python found. Install Python 3.11 or newer from python.org." "Red"
        exit 1
    }
    $Python = $candidates[0]
}

$pyVersion = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Say "Python        $pyVersion  ($Python)"

$arch = & $Python -c "import platform; print(platform.machine())"
$isArm = $arch -match "ARM64|aarch64"
if ($isArm) {
    Say "Architecture  $arch  - Snapdragon path" "Green"
} else {
    Say "Architecture  $arch  - x86 path (CPU fallback; NPU code still exercised)" "Yellow"
}

# -- 2. virtual environment ---------------------------------------------------

$venv = Join-Path $root ".venv"
if (-not (Test-Path $venv)) {
    Say "Creating virtual environment..."
    & $Python -m venv $venv
}
$venvPy = Join-Path $venv "Scripts\python.exe"

Say "Upgrading pip..."
& $venvPy -m pip install --quiet --upgrade pip setuptools wheel

# -- 3. dependencies ----------------------------------------------------------

Say "Installing dependencies..."
& $venvPy -m pip install --quiet -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Say "Dependency install failed." "Red"
    exit 1
}

if ($Dev) {
    Say "Installing dev extras..."
    & $venvPy -m pip install --quiet -r (Join-Path $root "requirements-dev.txt")
}

# onnxruntime-genai powers the glossary LLM. It is optional: without it the
# glossary falls back to a seeded heuristic rather than failing.
Say "Installing onnxruntime-genai (optional, for the glossary LLM)..."
& $venvPy -m pip install --quiet onnxruntime-genai 2>$null
if ($LASTEXITCODE -ne 0) {
    Say "  not available for this platform - glossary will use heuristics" "DarkYellow"
}

# -- 4. report what will actually execute -------------------------------------

Write-Host ""
Say "Checking the execution provider..." "White"
& $venvPy -m sahaay --device

# -- 5. models ----------------------------------------------------------------

if ($Models) {
    Write-Host ""
    Say "Downloading models..." "White"
    & $venvPy (Join-Path $root "scripts\download_models.py") --auto
} else {
    Write-Host ""
    Say "Models not downloaded. When you are ready:" "DarkGray"
    Say "  .\.venv\Scripts\python.exe scripts\download_models.py --auto" "DarkGray"
    Say "Sahaay runs without them - try:  .\run.bat --mock" "DarkGray"
}

Write-Host ""
Say "Done. Start Sahaay with:  .\run.bat" "Green"
Write-Host ""
