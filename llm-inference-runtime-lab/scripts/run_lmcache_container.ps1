# Start vLLM + LMCache (in-process) for Phase 7 smoke test.
# From llm-inference-runtime-lab/:
#   docker build -f docker/Dockerfile.lmcache -t llm-lab-vllm:lmcache .
#   .\scripts\run_lmcache_container.ps1
#
# LMCache flags are baked into the image CMD (avoids PowerShell JSON quoting).

$Image = "llm-lab-vllm:lmcache"
$Name = "llm-lab-lmcache"
$HfCache = Join-Path $env:USERPROFILE ".cache\huggingface"

docker image inspect $Image 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Image missing. Build first:"
    Write-Host "  docker build -f docker/Dockerfile.lmcache -t $Image ."
    exit 1
}

cmd /c "docker rm -f $Name >nul 2>nul"

Write-Host "Starting $Name from $Image (LMCache flags from image CMD) ..."
docker run --rm -d `
  --name $Name `
  --gpus all `
  --ipc=host `
  -p 8000:8000 `
  -v "${HfCache}:/root/.cache/huggingface" `
  $Image

if ($LASTEXITCODE -ne 0) {
    Write-Host "docker run failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Follow logs (look for LMCache Stored / Retrieved / hit):"
Write-Host "  docker logs -f $Name"
Write-Host ""
Write-Host "When healthy, in another terminal:"
Write-Host "  python scripts\smoke_lmcache.py"
Write-Host ""
Write-Host "Stop:"
Write-Host "  docker stop $Name"
