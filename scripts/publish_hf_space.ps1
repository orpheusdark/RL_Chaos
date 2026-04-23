param(
    [Parameter(Mandatory = $true)]
    [string]$HFUsername,

    [Parameter(Mandatory = $false)]
    [string]$SpaceName = "chaosops",

    [Parameter(Mandatory = $false)]
    [string]$HFToken = ""
)

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

if (-not [string]::IsNullOrWhiteSpace($HFToken)) {
    hf auth login --token $HFToken
}

$who = hf auth whoami
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face login required. Run: hf auth login"
}

hf repo create "$SpaceName" --type space --space-sdk docker --exist-ok
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create or access Hugging Face Space repo."
}

$repoUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"

if ((git remote) -contains "hf") {
    git remote remove hf
}

git remote add hf $repoUrl

git add .
$hasChanges = (git status --porcelain)
if (-not [string]::IsNullOrWhiteSpace($hasChanges)) {
    git commit -m "Prepare Hugging Face Space deployment"
}

git push hf main

Write-Host "Published to: $repoUrl"
