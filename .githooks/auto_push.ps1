# Auto-push script for Windows (PowerShell)
# Pushes to both GitHub (origin) and HuggingFace (hf)

Write-Host "Auto-pushing to both remotes..."

$gitDir = git rev-parse --show-toplevel
Set-Location $gitDir

# Push to GitHub
Write-Host "Pushing to GitHub (origin)..."
git push origin main
if ($LASTEXITCODE -ne 0) {
	Write-Error "Failed to push to origin"
	exit $LASTEXITCODE
}

# Push to HuggingFace Space  
Write-Host "Pushing to HuggingFace Space..."
git push hf main
if ($LASTEXITCODE -ne 0) {
	Write-Error "Failed to push to hf"
	exit $LASTEXITCODE
}

Write-Host "Auto-push complete"
