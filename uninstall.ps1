# LlamaForge uninstaller (Windows). Run from Apps & Features, or:
#   powershell -ExecutionPolicy Bypass -File uninstall.ps1
# Removes the app, its private Python and the llama.cpp builds it downloaded.
# Your settings and downloaded models stay unless you say otherwise.
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $env:TEMP

Write-Host "Uninstalling LlamaForge from $here" -ForegroundColor Yellow
if ((Test-Path (Join-Path $here "config.json")) -and -not $env:LLAMAFORGE_NO_STOP) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here "stop.ps1") *> $null
}

foreach ($lnk in (Join-Path ([Environment]::GetFolderPath("Programs")) "LlamaForge.lnk"),
                 (Join-Path ([Environment]::GetFolderPath("Desktop")) "LlamaForge.lnk")) {
  Remove-Item -Force $lnk -ErrorAction SilentlyContinue
}
Remove-Item -Recurse -Force "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LlamaForge" -ErrorAction SilentlyContinue

$all = Read-Host "Also delete your settings and the models LlamaForge downloaded into $here? (y/N)"
if ($all -eq "y") {
  Remove-Item -Recurse -Force $here -ErrorAction SilentlyContinue
} else {
  # Only files the installer put there (listed in its manifest), plus what it
  # downloaded for itself. Your config.json, models.ini and models\ stay.
  $manifest = Join-Path $here ".lf-files.json"
  if (Test-Path $manifest) {
    foreach ($rel in (Get-Content $manifest -Raw | ConvertFrom-Json).files) {
      if ($rel -match '(^|/)\.\.(/|$)' -or $rel -match '^[/\\]|:') { continue }
      Remove-Item -Force (Join-Path $here $rel) -ErrorAction SilentlyContinue
    }
  }
  foreach ($d in "python", "engines", "logs", ".lf-files.json", "stats.json") {
    Remove-Item -Recurse -Force (Join-Path $here $d) -ErrorAction SilentlyContinue
  }
  Get-ChildItem $here -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  # drop the now-empty code folders
  Get-ChildItem $here -Recurse -Directory | Sort-Object { $_.FullName.Length } -Descending |
    Where-Object { -not (Get-ChildItem $_.FullName -Force) } | Remove-Item -Force -ErrorAction SilentlyContinue
  Write-Host "Kept your settings and models in $here"
}
Write-Host "LlamaForge is uninstalled." -ForegroundColor Green
Start-Sleep -Seconds 3
