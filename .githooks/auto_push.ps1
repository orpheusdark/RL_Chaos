# Auto-push script for Windows (PowerShell)
# Pushes current branch to GitHub (origin) and Hugging Face (hf) if configured.

$ErrorActionPreference = "Stop"

$gitDir = git rev-parse --show-toplevel
Set-Location $gitDir

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ([string]::IsNullOrWhiteSpace($branch) -or $branch -eq "HEAD") {
	Write-Error "Cannot auto-push in detached HEAD state."
	exit 1
}

$remotes = @(git remote)
$pushedAny = $false

if ($remotes -contains "origin") {
	Write-Host "Pushing $branch to origin..."
	git push origin $branch
	if ($LASTEXITCODE -eq 0) {
		$pushedAny = $true
	} else {
		Write-Warning "Failed to push to origin"
	}
} else {
	Write-Host "Skipping origin push (remote not configured)."
}

if ($remotes -contains "hf") {
	# HF Space push uses a synthetic snapshot commit so local history issues
	# (for example, previously tracked binaries) do not block deployment.
	Write-Host "Pushing snapshot of $branch to hf/main..."
	git fetch hf main --quiet
	$tree = (git rev-parse "HEAD^{tree}").Trim()
	$parent = ""
	try {
		$parent = (git rev-parse "hf/main" 2>$null).Trim()
	} catch {
		$parent = ""
	}

	$message = "HF sync snapshot from $branch at $(Get-Date -Format s)"
	if ([string]::IsNullOrWhiteSpace($parent)) {
		$snapshotCommit = ($message | git commit-tree $tree).Trim()
	} else {
		$snapshotCommit = ($message | git commit-tree $tree -p $parent).Trim()
	}

	git push hf "$snapshotCommit`:main"
	if ($LASTEXITCODE -eq 0) {
		$pushedAny = $true
	} else {
		Write-Warning "Failed to push to hf"
	}
} else {
	Write-Host "Skipping hf push (remote not configured)."
}

if (-not $pushedAny) {
	Write-Error "Auto-push failed: no remote push succeeded."
	exit 1
}

Write-Host "Auto-push complete"
