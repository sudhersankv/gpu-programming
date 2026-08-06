# Run a kernel file via the LeetGPU CLI (remote compile/execute).
# Usage:
#   .\scripts\leetgpu_run.ps1 leetgpu\cuda\01_hello_cuda\hello_cuda.cu
#   .\scripts\leetgpu_run.ps1 leetgpu\cuda\01_hello_cuda\hello_cuda.cu -Mode cycle
#   .\scripts\leetgpu_run.ps1 path\to\kernel.py -Gpu "NVIDIA TESLA T4"

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Source,

    # Common values: functional, cycle-accurate (depends on CLI version)
    [string]$Mode = "",

    [string]$Gpu = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command leetgpu -ErrorAction SilentlyContinue)) {
    Write-Error @"
leetgpu CLI not found on PATH.

Install (Windows PowerShell):
  Invoke-WebRequest -Uri https://cli.leetgpu.com/install.ps1 -OutFile install.ps1; ./install.ps1

Docs: https://leetgpu.com/cli
"@
}

if (-not (Test-Path $Source)) {
    Write-Error "Source not found: $Source"
}

$SourceFull = (Resolve-Path $Source).Path
$argsList = @("run", $SourceFull)

if ($Mode) {
    $argsList += @("--mode", $Mode)
}
if ($Gpu) {
    $argsList += @("--gpu", $Gpu)
}

Write-Host "leetgpu $($argsList -join ' ')"
& leetgpu @argsList
exit $LASTEXITCODE
