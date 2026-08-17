# Start the FastAPI backend.
#
# HF_HOME must be a real process environment variable, not an entry in .env:
# pydantic-settings reads .env into the Settings object only, it does not
# export to os.environ — and `hf_hub_download` (app/embedding/onnx_embedder.py)
# reads the environment directly. Setting it in .env looks correct and silently
# re-downloads the model to ~/.cache/huggingface instead.
#
# This mirrors the VM's voicerag-api.service, so local and deployed behave the
# same way.

param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:HF_HOME = Join-Path $PSScriptRoot "data\hf_cache"
$env:OMP_NUM_THREADS = "4"
$env:PYTHONIOENCODING = "utf-8"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No venv. Run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

Write-Host "HF_HOME  = $env:HF_HOME"
Write-Host "listening on http://${BindHost}:$Port"

# --workers 1 is not optional: each worker loads its own full copy of every
# index, so N workers costs N x the memory for no gain on a 4-core box.
& $python -m uvicorn app.main:app --host $BindHost --port $Port --workers 1
