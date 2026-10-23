<#
.SYNOPSIS
  Builds Kreate's MSIX package for the Microsoft Store.

.DESCRIPTION
  Builds kreate.exe and kreate-mcp.exe in release, lays them out with the
  manifest and the tile images, and packs them with MakeAppx into
  dist\Kreate_<version>_x64.msix.

  Each run raises the version first: the last number of the workspace
  version in Cargo.toml goes up by one (0.3.0 -> 0.3.1), so every package
  made is newer than the one before, as the Store requires of an upload,
  and the application reports the same number. A failed build puts the
  old version back. -NoBump builds the version as it stands.

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
    [switch]$SkipBuild,
    [switch]$NoBump
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here '..\..')

# The workspace version, raised by one unless -NoBump. Read and written as
# the file's own text, so nothing but the number changes: no byte-order
# mark, the same line endings.
$cargoPath = Join-Path $repo 'Cargo.toml'
$noBom = New-Object System.Text.UTF8Encoding($false)
$original = [System.IO.File]::ReadAllText($cargoPath)
$pattern = '(?m)^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")'
$found = [regex]::Match($original, $pattern)
if (-not $found.Success) { throw 'no version in Cargo.toml' }
$major = [int]$found.Groups[2].Value
$minor = [int]$found.Groups[3].Value
$patch = [int]$found.Groups[4].Value
$bumped = $false
if (-not $NoBump) {
    $patch += 1
    $line = $found.Groups[1].Value + "$major.$minor.$patch" + $found.Groups[5].Value
    $text = $original.Substring(0, $found.Index) + $line +
        $original.Substring($found.Index + $found.Length)
    [System.IO.File]::WriteAllText($cargoPath, $text, $noBom)
    $bumped = $true
}
# The Store wants four numbers and the fourth zero: 0.3.1 -> 0.3.1.0.
$version = "$major.$minor.$patch.0"
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
        if ($built -ne 0) {
            # Nothing was made at the new number: it goes back.
            if ($bumped) { [System.IO.File]::WriteAllText($cargoPath, $original, $noBom) }
            throw 'cargo build failed'
        }
    }

    $layout = Join-Path $repo 'target\msix\layout'
    if (Test-Path $layout) { Remove-Item $layout -Recurse -Force }
    New-Item -ItemType Directory -Force (Join-Path $layout 'Assets') | Out-Null
    Copy-Item 'target\release\kreate.exe' $layout
    Copy-Item 'target\release\kreate-mcp.exe' $layout
    # DirectX's shader compiler, from the same SDK: with it the first frame
    # takes seconds, without it (FXC) most of a minute.
    foreach ($dll in 'dxcompiler.dll', 'dxil.dll') {
        $from = Join-Path $bin.FullName "x64\$dll"
        if (Test-Path $from) { Copy-Item $from $layout }
        else { Write-Warning "$dll not in the SDK: the package compiles shaders with FXC" }
    }
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
