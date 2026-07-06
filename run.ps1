# LlamaForge one-click runner.
# Reads config.json, starts the llama.cpp router + the LlamaForge backend,
# then opens the dashboard in your browser. Safe to run repeatedly.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg  = Get-Content (Join-Path $here "config.json") -Raw | ConvertFrom-Json

function Listening($port){ [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }

# 1. llama.cpp router (only if not already up)
if (-not (Listening $cfg.router_port)) {
  if (Test-Path $cfg.server_bin) {
    $args = @("--models-preset", $cfg.models_ini, "--models-max", "1", "--offline",
              "--host", "127.0.0.1", "--port", "$($cfg.router_port)", "--metrics")
    Start-Process -FilePath $cfg.server_bin -ArgumentList $args -WindowStyle Hidden
    Write-Host "started llama.cpp router on port $($cfg.router_port)"
  } else {
    Write-Host "server_bin not found ($($cfg.server_bin)) - open the dashboard Build tab to build llama.cpp first." -ForegroundColor Yellow
  }
}

# 2. LlamaForge backend (dashboard)
if (-not (Listening $cfg.panel_port)) {
  Start-Process -FilePath "python" -ArgumentList (Join-Path $here "backend\server.py") `
                -WorkingDirectory (Join-Path $here "backend") -WindowStyle Hidden
  Write-Host "started LlamaForge dashboard on port $($cfg.panel_port)"
}

# 3. open the dashboard
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:$($cfg.panel_port)/"
