# LlamaForge uninstaller (Windows). Run from Apps & Features, or:
#   powershell -ExecutionPolicy Bypass -File uninstall.ps1
# Removes the app, its private Python and the llama.cpp builds it downloaded.
# Your settings and downloaded models stay unless you say otherwise.
# Only a folder the installer made (it has .lf-files.json) is touched; what
# gets deleted is decided by backend\appinstall.py, never a blind recursive delete.
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $env:TEMP

if (-not (Test-Path (Join-Path $here ".lf-files.json"))) {
  Write-Host "$here was not made by the LlamaForge installer (no .lf-files.json) - nothing removed." -ForegroundColor Red
  Write-Host "For a git checkout, stop it with stop.ps1 and delete the folder yourself." -ForegroundColor Red
  Start-Sleep -Seconds 3
  exit 1
}

# Same order as run.ps1: the installer's private Python, then py, then python.
$python = $null
$candidates = @((Join-Path $here "python\python.exe"))
foreach ($name in "py", "python") {
  $cmd = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) { $candidates += $cmd.Source }
}
foreach ($candidate in $candidates) {
  if (-not (Test-Path $candidate)) { continue }
  & $candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
  if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
}
if (-not $python) {
  Write-Host "A working Python 3.10+ interpreter is required to uninstall (tried the bundled one, 'py' and 'python')." -ForegroundColor Red
  Start-Sleep -Seconds 3
  exit 1
}

Write-Host "Uninstalling LlamaForge from $here" -ForegroundColor Yellow
if ((Test-Path (Join-Path $here "config.json")) -and -not $env:LLAMAFORGE_NO_STOP) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "stop.ps1") *> $null
}

$all = Read-Host "Also delete your settings and the models LlamaForge downloaded into $here? (y/N)"
$uninstallArgs = @((Join-Path $here "backend\appinstall.py"), "--uninstall", $here)
if ($all -eq "y") { $uninstallArgs += "--all" }
& $python @uninstallArgs
if ($LASTEXITCODE -ne 0) {
  Write-Host "Uninstall stopped; nothing else was removed." -ForegroundColor Red
  Start-Sleep -Seconds 3
  exit 1
}

foreach ($lnk in (Join-Path ([Environment]::GetFolderPath("Programs")) "LlamaForge.lnk"),
                 (Join-Path ([Environment]::GetFolderPath("Desktop")) "LlamaForge.lnk")) {
  Remove-Item -Force $lnk -ErrorAction SilentlyContinue
}
Remove-Item -Recurse -Force "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LlamaForge" -ErrorAction SilentlyContinue
# The private Python ran the step above, so it goes last.
Remove-Item -Recurse -Force (Join-Path $here "python") -ErrorAction SilentlyContinue
if ((Test-Path $here) -and -not (Get-ChildItem $here -Force)) {
  Remove-Item -Force $here -ErrorAction SilentlyContinue
}
if ($all -ne "y") { Write-Host "Kept your settings and models in $here" }
Write-Host "LlamaForge is uninstalled." -ForegroundColor Green
Start-Sleep -Seconds 3
