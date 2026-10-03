# LlamaForge one-line installer for Windows.
#
#   irm https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/install.ps1 | iex
#
# No git, no admin, no build tools. It:
#   1. finds Python 3.10+, or downloads python.org's embeddable Python
#      (~11 MB, SHA-256 pinned) into the install dir - nothing system-wide
#   2. downloads the latest LlamaForge release and lays it over the install
#      dir (backend\appinstall.py: your config, models and engines are kept)
#   3. adds Start menu + desktop shortcuts and an Apps & Features entry
#   4. starts LlamaForge; the panel then offers the official llama.cpp build
#      for your GPU in one click
# Re-run it any time to update.
#
# Options (environment variables):
#   LLAMAFORGE_HOME        install dir (default %LOCALAPPDATA%\LlamaForge)
#   LLAMAFORGE_REF         release tag or branch (default: latest release)
#   LLAMAFORGE_ARCHIVE     install from a local .zip instead of downloading
#   LLAMAFORGE_NO_LAUNCH   don't start LlamaForge at the end
#   LLAMAFORGE_NO_SHORTCUTS  skip shortcuts and the Apps & Features entry
#   LLAMAFORGE_NO_STOP     don't stop a running copy before updating it
#   LLAMAFORGE_FORCE_EMBED_PYTHON  use the private Python even if one exists

function Install-LlamaForge {
  $ErrorActionPreference = "Stop"
  $ProgressPreference = "SilentlyContinue"      # PS 5.1's progress bar slows downloads 10x
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072  # TLS 1.2

  $repo = "dadwritestech/LlamaForge"
  $home_ = if ($env:LLAMAFORGE_HOME) { $env:LLAMAFORGE_HOME } else { Join-Path $env:LOCALAPPDATA "LlamaForge" }
  $pyVer = "3.12.10"
  # SHA-256 of python.org's embeddable zips (MD5 cross-checked against python.org)
  $pyZips = @{
    "AMD64" = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
    "ARM64" = "3065efc3d382d1cda66757ac71ade11904fa6e350f5a97eb74811acd71ba5532"
  }

  Write-Host ""
  Write-Host "  LlamaForge installer" -ForegroundColor Yellow
  Write-Host "  -> $home_"
  Write-Host ""
  New-Item -ItemType Directory -Force -Path $home_ | Out-Null
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("lf-install-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null

  try {
    # ---- 1. Python -------------------------------------------------------
    function Test-Py($exe) {
      try {
        & $exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        return $LASTEXITCODE -eq 0
      } catch { return $false }
    }
    $py = $null
    $bundled = Join-Path $home_ "python\python.exe"
    if (Test-Path $bundled) {
      if (Test-Py $bundled) { $py = $bundled }
    }
    if (-not $py -and -not $env:LLAMAFORGE_FORCE_EMBED_PYTHON) {
      foreach ($name in "py", "python", "python3") {
        $c = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($c -and (Test-Py $c.Source)) { $py = $c.Source; break }   # skips the Store shim
      }
    }
    if ($py) {
      Write-Host "  [1/4] Python: $py"
    } else {
      $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
      if (-not $pyZips.ContainsKey($arch)) { throw "No embeddable Python for $arch. Install Python 3.10+ from python.org and re-run." }
      $zipName = "python-$pyVer-embed-$($arch.ToLower()).zip"
      Write-Host "  [1/4] Python: downloading $zipName (private copy, ~11 MB)"
      $zip = Join-Path $tmp $zipName
      Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/$pyVer/$zipName" -OutFile $zip
      $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
      if ($got -ne $pyZips[$arch]) { throw "Python download failed its SHA-256 check (got $got). Nothing was installed." }
      $pyDir = Join-Path $home_ "python"
      if (Test-Path $pyDir) { Remove-Item -Recurse -Force $pyDir }
      Expand-Archive -Path $zip -DestinationPath $pyDir
      # The embeddable build ignores PYTHONPATH and the script's own folder;
      # its ._pth file is the whole sys.path. Add the backend to it.
      $pth = Get-ChildItem $pyDir -Filter "python*._pth" | Select-Object -First 1
      $stdlibZip = (Get-ChildItem $pyDir -Filter "python3*.zip" | Select-Object -First 1).Name
      Set-Content -Path $pth.FullName -Encoding ascii -Value @($stdlibZip, ".", "..\backend", "import site")
      $py = $bundled
      if (-not (Test-Py $py)) { throw "The private Python did not start." }
    }

    # ---- 2. LlamaForge ---------------------------------------------------
    if ($env:LLAMAFORGE_ARCHIVE) {
      $archive = $env:LLAMAFORGE_ARCHIVE
      $version = if ($env:LLAMAFORGE_REF) { $env:LLAMAFORGE_REF } else { "local" }
      Write-Host "  [2/4] LlamaForge: from $archive"
    } else {
      $ref = $env:LLAMAFORGE_REF
      # The latest release's tag, from the redirect github.com/<repo>/releases/latest
      # answers with - the API is rate-limited (60/h per IP, shared behind NAT/VPN),
      # so it is only the second try. Never fall back to master: a copy installed
      # from a branch records no version and is never offered an update again.
      if (-not $ref) {
        try {
          $q = [System.Net.HttpWebRequest]::Create("https://github.com/$repo/releases/latest")
          $q.Method = "HEAD"; $q.AllowAutoRedirect = $false; $q.UserAgent = "LlamaForge-installer"
          $r = $q.GetResponse()
          if ($r.Headers["Location"] -match '/releases/tag/([^/?#]+)$') { $ref = $Matches[1] }
          $r.Close()
        } catch { }
      }
      if (-not $ref) {
        try {
          $ref = (Invoke-RestMethod -UseBasicParsing -Headers @{ "User-Agent" = "LlamaForge-installer" } `
                    -Uri "https://api.github.com/repos/$repo/releases/latest").tag_name
        } catch { }
      }
      if (-not $ref) {
        throw "Could not find the latest LlamaForge release (GitHub unreachable or rate-limited). Retry in a few minutes, or pin one: `$env:LLAMAFORGE_REF = 'v0.15.0'"
      }
      $kind = if ($ref -match '^v\d') { "tags" } else { "heads" }
      $version = $ref
      $archive = Join-Path $tmp "llamaforge.zip"
      Write-Host "  [2/4] LlamaForge: downloading $ref"
      Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$repo/archive/refs/$kind/$ref.zip" -OutFile $archive
    }
    $src = Join-Path $tmp "src"
    Expand-Archive -Path $archive -DestinationPath $src
    $stop = Join-Path $home_ "stop.ps1"
    if ((Test-Path $stop) -and (Test-Path (Join-Path $home_ "config.json")) -and -not $env:LLAMAFORGE_NO_STOP) {
      Write-Host "        stopping the running copy to update it"
      try { & powershell -NoProfile -ExecutionPolicy Bypass -File $stop *> $null } catch { }
    }
    $installer = Get-ChildItem -Path $src -Recurse -Filter "appinstall.py" |
                 Where-Object { $_.Directory.Name -eq "backend" } | Select-Object -First 1
    if (-not $installer) { throw "That archive is not a LlamaForge release." }
    & $py $installer.FullName --from $src --to $home_ --version $version
    if ($LASTEXITCODE -ne 0) { throw "Copying LlamaForge into $home_ failed." }

    # ---- 3. shortcuts + Apps & Features ----------------------------------
    $vbs = Join-Path $home_ "LlamaForge.vbs"
    $ico = Join-Path $home_ "web\icons\llamaforge.ico"
    if (-not $env:LLAMAFORGE_NO_SHORTCUTS) {
      Write-Host "  [3/4] Start menu + desktop shortcuts"
      $shell = New-Object -ComObject WScript.Shell
      $startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "LlamaForge.lnk"
      $desktop = Join-Path ([Environment]::GetFolderPath("Desktop")) "LlamaForge.lnk"
      foreach ($path in $startMenu, $desktop) {
        $lnk = $shell.CreateShortcut($path)
        $lnk.TargetPath = Join-Path $env:WINDIR "System32\wscript.exe"
        $lnk.Arguments = "`"$vbs`""
        $lnk.WorkingDirectory = $home_
        $lnk.IconLocation = $ico
        $lnk.Description = "Local LLMs on llama.cpp"
        $lnk.Save()
      }
      $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\LlamaForge"
      New-Item -Path $key -Force | Out-Null
      $uninst = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $home_ 'uninstall.ps1')`""
      $vals = @{ DisplayName = "LlamaForge"; DisplayVersion = $version.TrimStart("v");
                 Publisher = "dadwritestech"; InstallLocation = $home_; DisplayIcon = $ico;
                 UninstallString = $uninst; URLInfoAbout = "https://github.com/$repo" }
      foreach ($k in $vals.Keys) { Set-ItemProperty -Path $key -Name $k -Value $vals[$k] }
      Set-ItemProperty -Path $key -Name NoModify -Value 1 -Type DWord
      Set-ItemProperty -Path $key -Name NoRepair -Value 1 -Type DWord
    } else {
      Write-Host "  [3/4] shortcuts skipped"
    }

    # ---- 4. launch -------------------------------------------------------
    Write-Host ""
    Write-Host "  LlamaForge $version is installed." -ForegroundColor Green
    Write-Host "  Useful? A GitHub star helps other people find it: https://github.com/dadwritestech/LlamaForge"
    if (-not $env:LLAMAFORGE_NO_LAUNCH) {
      Write-Host "  [4/4] Starting it - your browser will open the panel."
      Start-Process -FilePath (Join-Path $env:WINDIR "System32\wscript.exe") -ArgumentList "`"$vbs`""
    } else {
      Write-Host "  [4/4] Start it from the Start menu: LlamaForge"
    }
  } finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  }
}

Install-LlamaForge
