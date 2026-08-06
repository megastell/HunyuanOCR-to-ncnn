[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$SourceModelDirectory = "",
    [string]$Destination = "D:\hunyuanocr-recovery\phase4c\model-ntfs"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot "..\.."
}
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
if (-not $SourceModelDirectory) {
    $SourceModelDirectory = Join-Path $RepoRoot "artifacts"
}
$SourceModelDirectory = (Get-Item -LiteralPath $SourceModelDirectory).FullName
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Destination = (Get-Item -LiteralPath $Destination).FullName

$manifestPath = Join-Path $SourceModelDirectory "runtime_manifest.tsv"
$manifest = Get-Content -LiteralPath $manifestPath
if ($manifest.Count -lt 3 -or
    $manifest[0] -ne "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1") {
    throw "Unexpected runtime manifest header"
}
$declaredCount = [int](($manifest[1] -split "`t")[1])
$entries = $manifest | Select-Object -Skip 2
if ($entries.Count -ne $declaredCount) {
    throw "Manifest entry count differs from its header"
}

$copyTimer = [Diagnostics.Stopwatch]::StartNew()
$totalBytes = [int64]0
foreach ($line in $entries) {
    $columns = $line -split "`t"
    if ($columns.Count -ne 3) {
        throw "Invalid manifest entry: $line"
    }
    $relative = $columns[0].Replace("/", "\")
    $source = Join-Path $SourceModelDirectory $relative
    $target = Join-Path $Destination $relative
    $targetDirectory = Split-Path $target -Parent
    New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
    $totalBytes += [int64]$columns[1]
}
Copy-Item -LiteralPath $manifestPath `
    -Destination (Join-Path $Destination "runtime_manifest.tsv") -Force
$copyTimer.Stop()

$hashTimer = [Diagnostics.Stopwatch]::StartNew()
foreach ($line in $entries) {
    $columns = $line -split "`t"
    $target = Join-Path $Destination $columns[0].Replace("/", "\")
    $item = Get-Item -LiteralPath $target
    if ($item.Length -ne [int64]$columns[1]) {
        throw "Copied size mismatch: $target"
    }
    $actualHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $columns[2]) {
        throw "Copied SHA-256 mismatch: $target"
    }
}
$hashTimer.Stop()

$report = [ordered]@{
    format = "HUNYUANOCR_NCNN_NTFS_COPY_V1"
    source = $SourceModelDirectory
    destination = $Destination
    file_count = $declaredCount
    total_bytes = $totalBytes
    copy_seconds = [Math]::Round($copyTimer.Elapsed.TotalSeconds, 3)
    sha256_seconds = [Math]::Round($hashTimer.Elapsed.TotalSeconds, 3)
    status = "passed"
}
$reportPath = Join-Path (Split-Path $Destination -Parent) "model_ntfs_copy.json"
$json = $report | ConvertTo-Json -Depth 4
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($reportPath, $json + "`n", $utf8NoBom)
Write-Host "Manifest-selected model copy and SHA-256 verification passed."
Write-Host "Report: $reportPath"
