# LlamaForge first-run bootstrap (Windows).
# Gets the machine to the point where the GUI can take over: ensures Python +
# Git, fetches llama.cpp if absent, writes config.json, then launches the app.
# Heavy/consent-gated steps (compiler/CUDA install, the actual build) are done
# from the dashboard's Setup and Build tabs.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Have($cmd){ [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function TryInstall($id,$name){
  if (Have winget) {
    Write-Host "Installing $name via winget..." -ForegroundColor Cyan
    winget install --id $id -e --accept-source-agreements --accept-package-agreements
    return $true
  }
  return $false
}

Write-Host "=== LlamaForge bootstrap ===" -ForegroundColor Yellow

# --- Python (required to run the GUI backend) ---
if (-not (Have python)) {
  Write-Host "Python not found." -ForegroundColor Red
  $ans = Read-Host "Install Python 3.12 now? (y/n)"
  if ($ans -eq "y") { if (-not (TryInstall "Python.Python.3.12" "Python")) {
    Write-Host "No winget. Install Python manually: https://www.python.org/downloads/windows/"; exit 1 } }
  else { Write-Host "Python is required. Get it at https://www.python.org/downloads/windows/"; exit 1 }
  Write-Host "Re-open a new terminal so PATH updates, then run bootstrap again." ; exit 0
}

# --- Git (required to fetch/update llama.cpp) ---
if (-not (Have git)) {
  $ans = Read-Host "Git not found. Install it now? (y/n)"
  if ($ans -eq "y") { TryInstall "Git.Git" "Git" | Out-Null }
  else { Write-Host "Git recommended for building/updating llama.cpp: https://git-scm.com/download/win" }
}

# --- config.json ---
$cfgPath = Join-Path $here "config.json"
if (Test-Path $cfgPath) {
  $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
  Write-Host "Using existing config.json"
} else {
  $defaultSrc = Join-Path $here "llama.cpp"
  $src = $env:LLAMAFORGE_LLAMA_SRC
  if ([string]::IsNullOrWhiteSpace($src)) {
    $src = Read-Host "Existing llama.cpp source directory (Enter for $defaultSrc)"
  }
  if ([string]::IsNullOrWhiteSpace($src)) { $src = $defaultSrc }
  & python (Join-Path $here "backend\bootstrap_config.py") --config $cfgPath --repo-root $here --llama-src $src
  if ($LASTEXITCODE -ne 0) { throw "Could not create config.json" }
  $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
  Write-Host "Wrote config.json for $($cfg.llama_src)."
}

# --- fetch llama.cpp source if missing ---
if (-not (Test-Path (Join-Path $cfg.llama_src ".git"))) {
  $ans = Read-Host "llama.cpp source not found at $($cfg.llama_src). Clone it now (~fast)? (y/n)"
  if ($ans -eq "y" -and (Have git)) {
    git clone --depth 1 $cfg.git_remote $cfg.llama_src
  } else {
    Write-Host "Skipped. You can clone later or point config.json 'llama_src' at an existing checkout." -ForegroundColor Yellow
  }
}

# --- ensure a models.ini with a global section exists ---
if (-not (Test-Path $cfg.models_ini)) {
@"
version = 1

[*]
ctx-size = 8192
flash-attn = on
jinja = true
n-gpu-layers = 99
load-on-startup = false
"@ | Set-Content -Encoding ASCII $cfg.models_ini
  Write-Host "Created a starter models.ini"
}

Write-Host ""
Write-Host "Bootstrap done. Launching dashboard..." -ForegroundColor Green
Write-Host "In the app: Setup tab -> install any missing compiler/CUDA and scan drives;"
Write-Host "            Build tab -> build/update llama.cpp; Models tab -> tune + load."
& (Join-Path $here "run.ps1")
