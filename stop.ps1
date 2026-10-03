# LlamaForge one-click shutdown. Mirror of run.ps1.
# Stops the router this copy started, its model instances, the dashboard, and
# (if set up) its vLLM server - nothing else on the machine. The rules live in
# backend\procs.py. Safe to run repeatedly.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Same order as run.ps1: the installer's private Python, then py, then python.
$candidates = @((Join-Path $here "python\python.exe"))
foreach ($name in "py", "python") {
  $cmd = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) { $candidates += $cmd.Source }
}
foreach ($candidate in $candidates) {
  if (-not (Test-Path $candidate)) { continue }
  & $candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
  if ($LASTEXITCODE -ne 0) { continue }
  & $candidate (Join-Path $here "backend\procs.py") --stop $here
  exit $LASTEXITCODE
}
Write-Host "A working Python 3.10+ interpreter is required to stop LlamaForge (tried 'py' and 'python')." -ForegroundColor Red
exit 1
