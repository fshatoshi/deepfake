# Installe le bloc SSH vision-colab dans ~/.ssh/config
$ErrorActionPreference = "Stop"
$sshDir = Join-Path $env:USERPROFILE ".ssh"
$configPath = Join-Path $sshDir "config"
$blockPath = Join-Path $PSScriptRoot "ssh\config.vision-colab"

$marker = "Host vision-colab"
$block = (Get-Content $blockPath -Raw) -replace '(?m)^#.*\r?\n', ''

if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir | Out-Null
}

if (Test-Path $configPath) {
    $existing = Get-Content $configPath -Raw
    if ($existing -match [regex]::Escape($marker)) {
        Write-Host "Le bloc vision-colab est déjà présent dans $configPath"
        exit 0
    }
    Add-Content -Path $configPath -Value "`n$block"
} else {
    Set-Content -Path $configPath -Value $block.TrimEnd()
}

Write-Host "OK — config SSH : $configPath"
Write-Host "Test : ssh vision-colab"
