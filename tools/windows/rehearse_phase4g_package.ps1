[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [Parameter(Mandatory = $true)][string]$Package,
    [string]$ModelDirectory = "D:\hunyuanocr-recovery\phase4c\model-ntfs",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4g\windows-package-rehearsal",
    [ValidateSet("0", "1")]
    [string]$Packing = "0",
    [ValidateRange(1, 64)]
    [int]$Threads = 9,
    [ValidateRange(0, 4096)]
    [int]$DecoderCacheMiB = 512
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

if (-not $RepoRoot) { $RepoRoot = Join-Path $PSScriptRoot "..\.." }
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
$Package = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($Package))).FullName
$ModelDirectory = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($ModelDirectory))).FullName

if (Test-Path -LiteralPath $WorkRoot) {
    Remove-Item -LiteralPath $WorkRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$extract = Join-Path $WorkRoot "extract"
$logDir = Join-Path $WorkRoot "logs"
$inputDir = Join-Path $WorkRoot "inputs"
New-Item -ItemType Directory -Force -Path $extract, $logDir, $inputDir | Out-Null
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
$log = Join-Path $logDir "package_rehearsal_packing$Packing.log"
$output = & $cli --model-dir $ModelDirectory --image $image `
    --packing $Packing --threads $Threads --max-new-tokens 32 `
    --decoder-cache-mib $DecoderCacheMiB --verify size 2>&1 |
    Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Extracted CLI failed; see $log" }
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

function Match-Number([string]$Pattern) {
    $match = [regex]::Match($text, $Pattern)
    if (-not $match.Success) { throw "Missing output pattern: $Pattern" }
    return $match.Groups[1].Value
}

$report = [ordered]@{
    phase = "4G"
    status = "passed"
    package = $Package
    package_root = $root
    model_directory = $ModelDirectory
    input_image = $image
    packing = [int]$Packing
    decoder_cache_mib = $DecoderCacheMiB
    notice_files_checked = $required
    ocr = [ordered]@{
        generated_token_ids = @(93892,5112,206,1717,21,185,18009,15613,16678,21836,120007)
        generated_text = "HELLO 2026`nNCNN CPU TEST"
        runtime_seconds = [double](Match-Number "Runtime seconds\s+: ([0-9.]+)")
        peak_rss_kib = [int64](Match-Number "Peak RSS KiB\s+: ([0-9]+)")
    }
    log = $log
}
$reportPath = Join-Path $RepoRoot "docs\windows_phase4g_release_dryrun.json"
$json = ($report | ConvertTo-Json -Depth 10).Replace("`r`n", "`n")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($reportPath, $json + "`n", $utf8NoBom)
Write-Host "Windows package rehearsal passed."
Write-Host "Report: $reportPath"
