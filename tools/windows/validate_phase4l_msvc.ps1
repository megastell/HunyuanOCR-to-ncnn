[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$SourceModelDirectory = "\\wsl.localhost\Ubuntu-24.04\home\asus\hunyuanocr-recovery\phase4k\direct-staging-artifacts",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4l",
    [string]$NcnnSource = "",
    [string]$PythonExe = "",
    [ValidateRange(1, 64)]
    [int]$Jobs = 12,
    [ValidateRange(1, 64)]
    [int]$Threads = 9,
    [switch]$SkipCtest,
    [switch]$SkipCpack
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

if (-not $RepoRoot) { $RepoRoot = Join-Path $PSScriptRoot "..\.." }
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
$SourceModelDirectory = (
    Get-Item -LiteralPath ([IO.Path]::GetFullPath($SourceModelDirectory))
).FullName
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$WorkRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($WorkRoot))).FullName
$ModelDirectory = Join-Path $WorkRoot "model-ntfs"
$Logs = Join-Path $WorkRoot "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Invoke-LoggedScript {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Script
    )
    $log = Join-Path $Logs "$Label.log"
    & $Script 2>&1 | Tee-Object -FilePath $log
    $exitCode = 0
    if (Test-Path Variable:\LASTEXITCODE) {
        $exitCode = $LASTEXITCODE
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode; see $log"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Match-Required {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $match = [regex]::Match($Text, $Pattern)
    if (-not $match.Success) { throw "Missing output pattern: $Pattern" }
    return $match
}

function Invoke-PackageRehearsal {
    param(
        [Parameter(Mandatory = $true)][string]$Package,
        [Parameter(Mandatory = $true)][int]$Packing
    )
    $rehearsalRoot = Join-Path $WorkRoot "windows-package-rehearsal-packing$Packing"
    if (Test-Path -LiteralPath $rehearsalRoot) {
        Remove-Item -LiteralPath $rehearsalRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $rehearsalRoot | Out-Null
    $extract = Join-Path $rehearsalRoot "extract"
    $inputDir = Join-Path $rehearsalRoot "inputs"
    New-Item -ItemType Directory -Force -Path $extract, $inputDir | Out-Null
    Expand-Archive -LiteralPath $Package -DestinationPath $extract -Force
    $roots = @(Get-ChildItem -LiteralPath $extract -Directory)
    if ($roots.Count -ne 1) {
        throw "Expected one extracted package root in $extract"
    }
    $root = $roots[0].FullName
    $required = @(
        "share\doc\HunyuanOCR_ncnn\LICENSE",
        "share\doc\HunyuanOCR_ncnn\NOTICE",
        "share\doc\HunyuanOCR_ncnn\THIRD_PARTY_NOTICES.md",
        "share\doc\HunyuanOCR_ncnn\third_party\licenses\ncnn-LICENSE.txt",
        "share\doc\HunyuanOCR_ncnn\third_party\licenses\Tencent-HunyuanOCR-LICENSE.txt",
        "share\doc\HunyuanOCR_ncnn\third_party\licenses\stb_image-LICENSE.txt"
    )
    foreach ($relative in $required) {
        $path = Join-Path $root $relative
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Package is missing release notice file: $path"
        }
    }

    $image = Join-Path $inputDir "ocr_smoke_en.png"
    Copy-Item -LiteralPath (Join-Path $RepoRoot "tests\assets\ocr_smoke_en.png") `
        -Destination $image
    $cli = Join-Path $root "bin\hunyuanocr_cli.exe"
    if (-not (Test-Path -LiteralPath $cli)) { throw "Extracted CLI not found: $cli" }
    $log = Join-Path $Logs "windows_package_rehearsal_packing$Packing.log"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $output = & $cli --model-dir $ModelDirectory --image $image `
        --packing $Packing --threads $Threads --max-new-tokens 32 `
        --decoder-cache-mib 512 --verify size 2>&1 |
        Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    if ($exitCode -ne 0) { throw "Extracted CLI failed; see $log" }
    $text = $output -join "`n"
    $tokens = "93892 5112 206 1717 21 185 18009 15613 16678 21836 120007"
    $textMatch = [regex]::Match(
        $text,
        "Generated text:\r?\n(.*?)\r?\n\r?\nLoad seconds",
        [Text.RegularExpressions.RegexOptions]::Singleline)
    if ($text -notmatch "Generated tokens: $tokens" -or
        $text -notmatch "EOS reached\s+: true" -or
        -not $textMatch.Success -or
        $textMatch.Groups[1].Value -ne "HELLO 2026`nNCNN CPU TEST") {
        throw "Extracted package OCR output differs from expected smoke text"
    }
    return [ordered]@{
        status = "passed"
        package = $Package
        package_root = $root
        packing = $Packing
        log = $log
        wall_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
        generated_token_ids = @(93892,5112,206,1717,21,185,18009,15613,16678,21836,120007)
        generated_text = "HELLO 2026`nNCNN CPU TEST"
        runtime_seconds = [double](Match-Required "Runtime seconds\s+: ([0-9.]+)" $text).Groups[1].Value
        peak_rss_kib = [int64](Match-Required "Peak RSS KiB\s+: ([0-9]+)" $text).Groups[1].Value
    }
}

Invoke-LoggedScript "copy_manifest_model" {
    & (Join-Path $RepoRoot "tools\windows\copy_manifest_model.ps1") `
        -RepoRoot $RepoRoot `
        -SourceModelDirectory $SourceModelDirectory `
        -Destination $ModelDirectory
}
$copyReportPath = Join-Path $WorkRoot "model_ntfs_copy.json"

$windowsReport = Join-Path $RepoRoot "docs\windows_phase4l_release_validation.json"
Invoke-LoggedScript "windows_phase4l_release_validation" {
    & (Join-Path $RepoRoot "tools\windows\validate_phase4f_msvc.ps1") `
        -RepoRoot $RepoRoot `
        -NcnnSource $NcnnSource `
        -PythonExe $PythonExe `
        -WorkRoot (Join-Path $WorkRoot "windows-release-validation") `
        -ModelDirectory $ModelDirectory `
        -Phase "4L" `
        -ReportPath $windowsReport `
        -Jobs $Jobs `
        -Threads $Threads `
        -SkipCtest:$SkipCtest `
        -SkipCpack:$SkipCpack
}

$packageDir = Join-Path $WorkRoot "windows-release-validation\packages"
$packages = @(Get-ChildItem -LiteralPath $packageDir -Filter "*.zip" |
    Sort-Object LastWriteTime, Name)
if ($packages.Count -eq 0) { throw "Windows release validation did not produce a ZIP package" }
$package = $packages[-1].FullName
$packageRehearsals = @(
    Invoke-PackageRehearsal $package 0
    Invoke-PackageRehearsal $package 1
)

$copyReport = Get-Content -Raw -LiteralPath $copyReportPath | ConvertFrom-Json
$releaseReport = Get-Content -Raw -LiteralPath $windowsReport | ConvertFrom-Json
$summary = [ordered]@{
    phase = "4L"
    status = "passed"
    repo_root = $RepoRoot
    source_model_directory = $SourceModelDirectory
    ntfs_model_directory = $ModelDirectory
    work_root = $WorkRoot
    model_copy = $copyReport
    release_validation_report = $windowsReport
    ctest = $releaseReport.ctest
    packages = $releaseReport.packages
    package_rehearsals = $packageRehearsals
    model_manifest_sha256 = Get-Sha256 (Join-Path $ModelDirectory "runtime_manifest.tsv")
    model_compatibility_sha256 = Get-Sha256 (Join-Path $ModelDirectory "runtime_compatibility.tsv")
    persistent_log_root = $Logs
}
$summaryPath = Join-Path $RepoRoot "docs\windows_phase4l_reproduced_artifact_acceptance.json"
$json = ($summary | ConvertTo-Json -Depth 14).Replace("`r`n", "`n")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($summaryPath, $json + "`n", $utf8NoBom)
Write-Host "Native Windows Phase 4L reproduced artifact acceptance passed."
Write-Host "Report: $summaryPath"
