<#
.SYNOPSIS
  Builds Kreate's MSIX package for the Microsoft Store.

.DESCRIPTION
  Builds kreate.exe and kreate-mcp.exe in release, lays them out with the
  manifest and the tile images, and packs them with MakeAppx into
  dist\Kreate_<version>_x64.msix.

  The package is unsigned: the Store signs what is uploaded to it. To
  install it on this machine before uploading, pass -CertificatePath with a
  .pfx whose subject is the manifest's Publisher (and trust that
  certificate), and it is signed with SignTool.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\msix\build.ps1
#>
param(
    [string]$CertificatePath,
    [string]$CertificatePassword,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here '..\..')

# The Store wants four numbers and the fourth zero: 0.3.0 -> 0.3.0.0.
$cargo = Get-Content (Join-Path $repo 'Cargo.toml') -Raw
if ($cargo -notmatch '(?m)^version\s*=\s*"(\d+)\.(\d+)\.(\d+)') {
    throw 'no version in Cargo.toml'
}
$version = "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"
Write-Host "Kreate $version"

# The newest SDK that has MakeAppx.
$kits = 'C:\Program Files (x86)\Windows Kits\10\bin'
$bin = Get-ChildItem $kits -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'x64\makeappx.exe') } |
    Sort-Object { [version]$_.Name } | Select-Object -Last 1
if (-not $bin) { throw "MakeAppx not found under ${kits}: install the Windows SDK" }
$makeappx = Join-Path $bin.FullName 'x64\makeappx.exe'
$signtool = Join-Path $bin.FullName 'x64\signtool.exe'

Push-Location $repo
try {
    if (-not $SkipBuild) {
        # Cargo reports progress on stderr, which Windows PowerShell turns
        # into errors that stop the script: let it speak, and judge by the
        # exit code.
        $ErrorActionPreference = 'Continue'
        cargo build --release -p kreate -p kreate-mcp 2>&1 | ForEach-Object { "$_" }
        $built = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($built -ne 0) { throw 'cargo build failed' }
    }

    $layout = Join-Path $repo 'target\msix\layout'
    if (Test-Path $layout) { Remove-Item $layout -Recurse -Force }
    New-Item -ItemType Directory -Force (Join-Path $layout 'Assets') | Out-Null
    Copy-Item 'target\release\kreate.exe' $layout
    Copy-Item 'target\release\kreate-mcp.exe' $layout
    Copy-Item (Join-Path $here 'Assets\*') (Join-Path $layout 'Assets')
    (Get-Content (Join-Path $here 'AppxManifest.xml') -Raw).Replace('$VERSION$', $version) |
        Set-Content (Join-Path $layout 'AppxManifest.xml') -Encoding utf8

    $dist = Join-Path $repo 'dist'
    New-Item -ItemType Directory -Force $dist | Out-Null
    $msix = Join-Path $dist "Kreate_${version}_x64.msix"
    & $makeappx pack /o /d $layout /p $msix
    if ($LASTEXITCODE -ne 0) { throw 'MakeAppx failed' }

    if ($CertificatePath) {
        $sign = @('sign', '/fd', 'SHA256', '/f', $CertificatePath)
        if ($CertificatePassword) { $sign += @('/p', $CertificatePassword) }
        & $signtool @sign $msix
        if ($LASTEXITCODE -ne 0) { throw 'SignTool failed' }
    }
    Write-Host "Made $msix"
}
finally {
    Pop-Location
}
