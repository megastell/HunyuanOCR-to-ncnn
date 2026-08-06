[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4c",
    [string]$ModelDirectory = "",
    [ValidateRange(1, 64)]
    [int]$Threads = 9
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot "..\.."
}
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
if (-not $ModelDirectory) {
    $ModelDirectory = Join-Path $WorkRoot "model-ntfs"
}
$cli = Join-Path $WorkRoot "runtime-install\bin\hunyuanocr_cli.exe"
$manifestPath = Join-Path $ModelDirectory "runtime_manifest.tsv"
$expectedPath = Join-Path $RepoRoot "tests\assets\dynamic_ocr_expected.json"
foreach ($path in @($cli, $manifestPath, $expectedPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path not found: $path"
    }
}

$manifest = Get-Content -LiteralPath $manifestPath
if ($manifest[0] -ne "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1") {
    throw "Unexpected runtime manifest header"
}
$fileCount = [int](($manifest[1] -split "`t")[1])
if (($manifest.Count - 2) -ne $fileCount) {
    throw "Manifest file count mismatch"
}
$totalBytes = [int64]0
foreach ($line in $manifest | Select-Object -Skip 2) {
    $columns = $line -split "`t"
    if ($columns.Count -ne 3) { throw "Invalid manifest entry: $line" }
    $totalBytes += [int64]$columns[1]
}

function Read-Metric([string]$Text, [string]$Label) {
    $match = [regex]::Match($Text, "$Label\s+: ([0-9.]+)")
    if (-not $match.Success) { throw "Missing metric: $Label" }
    return [double]$match.Groups[1].Value
}

function Invoke-OcrCase {
    param(
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][int]$Packing,
        [Parameter(Mandatory = $true)][string]$Verification
    )
    $name = "$($Case.name)_packing$Packing"
    $log = Join-Path $WorkRoot "windows_${name}.log"
    $image = Join-Path $RepoRoot $Case.path.Replace("/", "\")
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $output = & $cli --model-dir $ModelDirectory --image $image `
        --packing $Packing --threads $Threads --max-new-tokens 32 `
        --verify $Verification 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    if ($exitCode -ne 0) {
        throw "$name failed with exit code $exitCode; see $log"
    }
    $text = $output -join "`n"
    $expectedTokens = ($Case.generated_token_ids -join " ")
    $expectedGrid = $Case.grid_thw -join ","
    $expectedSpan = $Case.image_token_span -join ","
    $textMatch = [regex]::Match(
        $text,
        "Generated text:\r?\n(.*?)\r?\n\r?\nLoad seconds",
        [Text.RegularExpressions.RegexOptions]::Singleline)
    if ($text -notmatch "Generated tokens: $expectedTokens" -or
        $text -notmatch "Image grid\s+: \[$expectedGrid\]" -or
        $text -notmatch "Image token span: \[$expectedSpan\)" -or
        $text -notmatch "Prefill length\s+: $($Case.sequence_length)" -or
        $text -notmatch "EOS reached\s+: true" -or
        -not $textMatch.Success -or
        $textMatch.Groups[1].Value -ne $Case.generated_text) {
        throw "$name did not match the exact PyTorch reference"
    }
    return [ordered]@{
        case = $Case.name
        packing = $Packing
        verification = $Verification
        grid_thw = $Case.grid_thw
        image_token_span = $Case.image_token_span
        prefill_length = $Case.sequence_length
        tokens = $Case.generated_token_ids
        text = $Case.generated_text
        load_seconds = Read-Metric $text "Load seconds"
        input_seconds = Read-Metric $text "Input seconds"
        prefill_seconds = Read-Metric $text "Prefill seconds"
        decode_seconds = Read-Metric $text "Decode seconds"
        runtime_seconds = Read-Metric $text "Runtime seconds"
        peak_rss_kib = [int64](Read-Metric $text "Peak RSS KiB")
        wall_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    }
}

$dynamic = Get-Content -Raw -LiteralPath $expectedPath | ConvertFrom-Json
$cases = @($dynamic.cases)
$cases += [pscustomobject]@{
    name = "ocr_smoke_en"
    path = "tests/assets/ocr_smoke_en.png"
    grid_thw = @(1, 22, 50)
    sequence_length = 313
    image_token_span = @(2, 290)
    generated_token_ids = @(93892,5112,206,1717,21,185,18009,15613,16678,21836,120007)
    generated_text = "HELLO 2026`nNCNN CPU TEST"
}
$results = @()
foreach ($case in $cases) {
    foreach ($packing in @(0, 1)) {
        $verification = if ($case.name -eq "ocr_smoke_en" -and $packing -eq 1) {
            "sha256"
        } else {
            "size"
        }
        $results += Invoke-OcrCase $case $packing $verification
        Write-Host "PASS $($case.name) packing=$packing"
    }
}

$os = Get-CimInstance Win32_OperatingSystem
$report = [ordered]@{
    phase = "4C"
    status = "passed"
    platform = [ordered]@{
        os = $os.Caption
        version = $os.Version
        architecture = $os.OSArchitecture
        cmake = (& cmake --version | Select-Object -First 1)
    }
    manifest = [ordered]@{
        format = "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1"
        file_count = $fileCount
        total_bytes = $totalBytes
        ntfs_directory = $ModelDirectory
        size_verification = "passed"
        cpp_sha256_verification = "passed"
    }
    results = $results
}
$reportPath = Join-Path $RepoRoot "docs\windows_phase4c_validation.json"
$json = $report | ConvertTo-Json -Depth 10
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    $reportPath,
    ($json -replace "`r`n", "`n") + "`n",
    $utf8NoBom)
Write-Host "Native Windows Phase 4C regression passed."
Write-Host "Report: $reportPath"
