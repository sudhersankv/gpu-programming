# Compile and run a CUDA .cu file with nvcc (Windows-friendly).
# Usage:
#   .\scripts\build_cu.ps1 lessons\01_hello_cuda\hello.cu
#   .\scripts\build_cu.ps1 lessons\01_hello_cuda\hello.cu -OutDir build -Run

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Source,

    [string]$OutDir = "build",

    [switch]$Run
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Error "nvcc not found on PATH. Install CUDA Toolkit or add ...\CUDA\v12.6\bin to PATH."
}

if (-not (Test-Path $Source)) {
    Write-Error "Source not found: $Source"
}

# nvcc on Windows needs MSVC (cl.exe). Load it if missing.
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "cl.exe not on PATH and vswhere not found. Install VS Build Tools (Desktop development with C++)."
    }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) {
        Write-Error "Visual Studio C++ tools not found. Install 'Desktop development with C++' / MSVC."
    }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        Write-Error "vcvars64.bat not found at $vcvars"
    }
    Write-Host "Loading MSVC via $vcvars"
    $envBatch = cmd /c "`"$vcvars`" >nul && set"
    foreach ($line in $envBatch -split "`r?`n") {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Write-Error "Failed to put cl.exe on PATH after loading vcvars64."
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$base = [System.IO.Path]::GetFileNameWithoutExtension($Source)
$outExe = Join-Path $OutDir ($base + ".exe")
$SourceFull = (Resolve-Path $Source).Path
$OutFull = Join-Path (Resolve-Path $OutDir).Path ($base + ".exe")

Write-Host "nvcc $Source -> $outExe"
nvcc $SourceFull -o $OutFull -O2 -std=c++17
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Built: $outExe"
if ($Run) {
    Write-Host "Running..."
    & $OutFull
    exit $LASTEXITCODE
}
