[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4d",
    [string]$ModelDirectory = "D:\hunyuanocr-recovery\phase4c\model-ntfs",
    [ValidateRange(1, 64)]
    [int]$Threads = 9,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

if (-not $RepoRoot) { $RepoRoot = Join-Path $PSScriptRoot "..\.." }
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
$cli = Join-Path $WorkRoot "runtime-install\bin\hunyuanocr_cli.exe"
$benchmark = Join-Path $WorkRoot "runtime-build\hunyuanocr_repeat_benchmark.exe"
$dynamicExpectedPath = Join-Path $RepoRoot "tests\assets\dynamic_ocr_expected.json"
$realExpectedPath = Join-Path $RepoRoot "tests\assets\real_ocr_expected.json"
foreach ($path in @($cli, $benchmark, $ModelDirectory, $dynamicExpectedPath, $realExpectedPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

function Invoke-OcrCase {
    param(
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][int]$Packing,
        [Parameter(Mandatory = $true)][int]$MaxTokens,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $image = Join-Path $RepoRoot $Case.path.Replace("/", "\")
    $log = Join-Path $WorkRoot "windows_${Name}_packing${Packing}.log"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    if ($Resume -and (Test-Path -LiteralPath $log)) {
        $output = Get-Content -LiteralPath $log
        $exitCode = 0
    } else {
        $output = & $cli --model-dir $ModelDirectory --image $image `
            --packing $Packing --threads $Threads --max-new-tokens $MaxTokens `
            --verify size 2>&1 | Tee-Object -FilePath $log
        $exitCode = $LASTEXITCODE
    }
    $timer.Stop()
    if ($exitCode -ne 0) { throw "$Name packing=$Packing failed; see $log" }
    $text = $output -join "`n"
    $tokens = $Case.generated_token_ids -join " "
    $textMatch = [regex]::Match(
        $text,
        "Generated text:\r?\n(.*?)\r?\n\r?\nLoad seconds",
        [Text.RegularExpressions.RegexOptions]::Singleline)
    if ($text -notmatch "Generated tokens: $tokens" -or
        $text -notmatch "EOS reached\s+: true" -or
        -not $textMatch.Success -or
        $textMatch.Groups[1].Value -ne $Case.generated_text) {
        throw "$Name packing=$Packing differs from the PyTorch reference"
    }
    $peak = [regex]::Match($text, "Peak RSS KiB\s+: ([0-9]+)")
    $runtime = [regex]::Match($text, "Runtime seconds\s+: ([0-9.]+)")
    return [ordered]@{
        case = $Name
        packing = $Packing
        tokens = $Case.generated_token_ids
        text = $Case.generated_text
        runtime_seconds = [double]$runtime.Groups[1].Value
        wall_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
        peak_rss_kib = [int64]$peak.Groups[1].Value
    }
}

function Invoke-RepeatBenchmark {
    param([Parameter(Mandatory = $true)][int]$Packing)
    $dynamic = Get-Content -Raw -LiteralPath $dynamicExpectedPath | ConvertFrom-Json
    $images = @($dynamic.cases | ForEach-Object {
        Join-Path $RepoRoot $_.path.Replace("/", "\")
    })
    $smoke = Join-Path $RepoRoot "tests\assets\ocr_smoke_en.png"
    $images += $smoke
    $log = Join-Path $WorkRoot "windows_repeat10_packing${Packing}.log"
    $output = & $benchmark $ModelDirectory $Packing $Threads 10 @images 2>&1 |
        Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Repeat benchmark failed; see $log" }
    $lines = @($output | Where-Object { $_ -match "^iteration=" })
    if ($lines.Count -ne 10) { throw "Repeat benchmark did not run 10 iterations" }
    $rss = @()
    foreach ($line in $lines) {
        $match = [regex]::Match($line, "current_rss_kib=([0-9]+)")
        if (-not $match.Success) { throw "Missing current RSS in $line" }
        $rss += [int64]$match.Groups[1].Value
    }
    $peakMatch = [regex]::Match(($output -join "`n"), "peak_rss_kib=([0-9]+)\s*$")
    $growth = $rss[-1] - $rss[1]
    if ($growth -gt 262144) {
        throw "Repeat RSS grew by more than 256 MiB after warmup: $growth KiB"
    }
    return [ordered]@{
        packing = $Packing
        iterations = 10
        current_rss_kib = $rss
        warmup_to_final_growth_kib = $growth
        peak_rss_kib = [int64]$peakMatch.Groups[1].Value
        status = "passed"
    }
}

$dynamic = Get-Content -Raw -LiteralPath $dynamicExpectedPath | ConvertFrom-Json
$real = Get-Content -Raw -LiteralPath $realExpectedPath | ConvertFrom-Json
$exactResults = @()
foreach ($case in $dynamic.cases) {
    foreach ($packing in @(0, 1)) {
        $exactResults += Invoke-OcrCase $case $packing 32 $case.name
    }
}
$smoke = [pscustomobject]@{
    name = "ocr_smoke_en"
    path = "tests/assets/ocr_smoke_en.png"
    generated_token_ids = @(93892,5112,206,1717,21,185,18009,15613,16678,21836,120007)
    generated_text = "HELLO 2026`nNCNN CPU TEST"
}
foreach ($packing in @(0, 1)) {
    $exactResults += Invoke-OcrCase $smoke $packing 32 $smoke.name
    $exactResults += Invoke-OcrCase $real.cases[0] $packing 256 $real.cases[0].name
}

$jpeg = Join-Path $RepoRoot "tests\assets\ocr_receipt_real.jpg"
$jpegLog = Join-Path $WorkRoot "windows_jpeg_compatibility.log"
if ($Resume -and (Test-Path -LiteralPath $jpegLog)) {
    $jpegOutput = Get-Content -LiteralPath $jpegLog
    $jpegExit = 0
} else {
    $jpegOutput = & $cli --model-dir $ModelDirectory --image $jpeg --packing 0 `
        --threads $Threads --max-new-tokens 256 --verify size 2>&1 |
        Tee-Object -FilePath $jpegLog
    $jpegExit = $LASTEXITCODE
}
if ($jpegExit -ne 0 -or ($jpegOutput -join "`n") -notmatch "EOS reached\s+: true") {
    throw "JPEG compatibility execution failed"
}

$largeJpeg = Join-Path $RepoRoot `
    "..\HunyuanOCR-official\HunyuanOCR_v1.0\assets\vis_document_23.jpg"
$limitLog = Join-Path $WorkRoot "windows_grid_limit.log"
if ($Resume -and (Test-Path -LiteralPath $limitLog)) {
    $limitOutput = Get-Content -LiteralPath $limitLog
    $limitExit = 1
} else {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $limitOutput = & $cli --model-dir $ModelDirectory --image $largeJpeg `
        --packing 0 --threads $Threads --verify size 2>&1 |
        Tee-Object -FilePath $limitLog
    $limitExit = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
}
if ($limitExit -eq 0 -or
    ($limitOutput -join "`n") -notmatch "exceeding the configured limit of 2048") {
    throw "Vision patch limit did not reject the large image clearly"
}

$repeatResults = @(
    Invoke-RepeatBenchmark 0
    Invoke-RepeatBenchmark 1
)
$report = [ordered]@{
    phase = "4D"
    status = "passed"
    platform = (Get-CimInstance Win32_OperatingSystem).Caption
    model_directory = $ModelDirectory
    exact_results = $exactResults
    jpeg_compatibility = [ordered]@{
        status = "passed"
        pyTorch_exact_parity = "not claimed; stb and Pillow JPEG pixels differ"
    }
    patch_limit = "passed"
    repeat_results = $repeatResults
}
$reportPath = Join-Path $RepoRoot "docs\windows_phase4d_validation.json"
$json = $report | ConvertTo-Json -Depth 12
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($reportPath, $json + "`n", $utf8NoBom)
Write-Host "Native Windows Phase 4D validation passed."
Write-Host "Report: $reportPath"
