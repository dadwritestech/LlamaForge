# LlamaForge one-click shutdown. Mirror of run.ps1.
# Stops the llama.cpp router, every model instance it spawned, and the
# LlamaForge dashboard backend. Safe to run repeatedly.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg  = Get-Content (Join-Path $here "config.json") -Raw | ConvertFrom-Json

function Kill-Port($port, $label) {
  $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
          Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($processId in $pids) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Write-Host "stopped $label (pid $processId on port $port)"
  }
}

# 1. dashboard backend (python backend\server.py on panel_port)
Kill-Port $cfg.panel_port "LlamaForge dashboard"

# 2. router on router_port
Kill-Port $cfg.router_port "llama.cpp router"

# 3. sweep any llama-server model instances the router spawned on random ports
$instances = Get-Process llama-server -ErrorAction SilentlyContinue
foreach ($p in $instances) {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  Write-Host "stopped model instance (pid $($p.Id))"
}

# 4. vLLM runs inside WSL - kill any `vllm serve` process there
$distro = $cfg.wsl_distro
try {
  if ($distro) { wsl.exe -d $distro -- bash -lc "pkill -f 'vllm serve' 2>/dev/null; true" }
  else         { wsl.exe -- bash -lc "pkill -f 'vllm serve' 2>/dev/null; true" }
  Write-Host "stopped any vLLM serve process in WSL"
} catch {
  Write-Host "WSL not available or no vLLM running" -ForegroundColor DarkGray
}

if (-not $instances -and -not (Get-NetTCPConnection -LocalPort $cfg.router_port,$cfg.panel_port -State Listen -ErrorAction SilentlyContinue)) {
  Write-Host "nothing was running." -ForegroundColor DarkGray
}
Write-Host "LlamaForge stopped." -ForegroundColor Green
