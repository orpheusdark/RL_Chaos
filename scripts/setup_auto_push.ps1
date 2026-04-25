param(
    [Parameter(Mandatory = $true)]
    [string]$HFUsername,

    [Parameter(Mandatory = $false)]
    [string]$SpaceName = "chaosops"
)

$ErrorActionPreference = "Stop"

git -C "$PSScriptRoot\.." config core.hooksPath .githooks
$repoUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"
if ((git -C "$PSScriptRoot\.." remote) -contains "hf") {
    git -C "$PSScriptRoot\.." remote set-url hf $repoUrl
} else {
    git -C "$PSScriptRoot\.." remote add hf $repoUrl
}
Write-Host "Auto push is enabled."
Write-Host "Hooks path: $(git -C "$PSScriptRoot\.." config --get core.hooksPath)"
Write-Host "HF remote: $(git -C "$PSScriptRoot\.." remote get-url hf)"
