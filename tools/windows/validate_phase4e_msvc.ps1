[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4e",
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
$decoder = Join-Path $WorkRoot "runtime-build\hunyuanocr_decode_image_rgb.exe"
$expectedPath = Join-Path $RepoRoot "tests\assets\real_ocr_expected.json"
$jpeg = Join-Path $RepoRoot "tests\assets\ocr_receipt_real.jpg"
$rgbContract = Join-Path $RepoRoot "tests\assets\ocr_receipt_real_stb.ppm"
foreach ($path in @($cli, $decoder, $ModelDirectory, $expectedPath, $jpeg, $rgbContract)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}

function Invoke-OcrCase {
    param(
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][int]$Packing,
        [Parameter(Mandatory = $true)][int]$BudgetMiB,
        [Parameter(Mandatory = $true)][int]$MaxTokens,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $image = Join-Path $RepoRoot $Case.path.Replace("/", "\")
    $log = Join-Path $WorkRoot `
        "windows_${Name}_packing${Packing}_cache${BudgetMiB}.log"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    if ($Resume -and (Test-Path -LiteralPath $log)) {
        $output = Get-Content -LiteralPath $log
        $exitCode = 0
    } else {
        $output = & $cli --model-dir $ModelDirectory --image $image `
            --packing $Packing --threads $Threads --max-new-tokens $MaxTokens `
            --decoder-cache-mib $BudgetMiB --verify size 2>&1 |
            Tee-Object -FilePath $log
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
        throw "$Name packing=$Packing cache=$BudgetMiB differs from PyTorch"
    }
    function Match-Number([string]$Pattern) {
        $match = [regex]::Match($text, $Pattern)
        if (-not $match.Success) { throw "Missing output pattern: $Pattern" }
        return $match.Groups[1].Value
    }
    return [ordered]@{
        case = $Name
        packing = $Packing
        decoder_cache_budget_mib = $BudgetMiB
        generated_token_count = $Case.generated_token_ids.Count
        runtime_seconds = [double](Match-Number "Runtime seconds\s+: ([0-9.]+)")
        decode_seconds = [double](Match-Number "Decode seconds\s+: ([0-9.]+)")
        peak_rss_kib = [int64](Match-Number "Peak RSS KiB\s+: ([0-9]+)")
        memory_cached_layers = [int](Match-Number "Memory layers\s+: ([0-9]+)")
        cache_estimated_mib = [int](Match-Number "Cache estimate\s+: ([0-9]+) MiB")
        wall_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
        status = "passed"
    }
}

$windowsRgb = Join-Path $WorkRoot "ocr_receipt_real_stb_windows.ppm"
$decodeLog = Join-Path $WorkRoot "windows_stb_rgb_contract.log"
& $decoder $jpeg $windowsRgb 2>&1 | Tee-Object -FilePath $decodeLog
if ($LASTEXITCODE -ne 0) { throw "Windows stb RGB export failed" }
$expectedRgbSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $rgbContract).Hash
$windowsRgbSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $windowsRgb).Hash
if ($windowsRgbSha -ne $expectedRgbSha) {
    throw "Windows and Linux stb RGB contract hashes differ"
}

$expected = Get-Content -Raw -LiteralPath $expectedPath | ConvertFrom-Json
$jpegCase = $expected.cases | Where-Object { $_.name -eq "ocr_receipt_real_jpeg" }
if ($null -eq $jpegCase) { throw "JPEG PyTorch reference case is missing" }
$smoke = [pscustomobject]@{
    path = "tests/assets/ocr_smoke_en.png"
    generated_token_ids = @(93892,5112,206,1717,21,185,18009,15613,16678,21836,120007)
    generated_text = "HELLO 2026`nNCNN CPU TEST"
}
$results = @()
foreach ($packing in @(0, 1)) {
    $results += Invoke-OcrCase $smoke $packing 512 32 "ocr_smoke_en"
}
foreach ($budget in @(0, 2048)) {
    foreach ($packing in @(0, 1)) {
        $results += Invoke-OcrCase `
            $jpegCase $packing $budget 256 "ocr_receipt_real_jpeg"
    }
}

$report = [ordered]@{
    phase = "4E"
    status = "passed"
    platform = (Get-CimInstance Win32_OperatingSystem).Caption
    model_directory = $ModelDirectory
    stb_rgb_contract = [ordered]@{
        status = "passed"
        sha256 = $windowsRgbSha.ToLowerInvariant()
        linux_windows_byte_identical = $true
    }
    results = $results
}
$reportPath = Join-Path $RepoRoot "docs\windows_phase4e_validation.json"
$json = ($report | ConvertTo-Json -Depth 10).Replace("`r`n", "`n")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($reportPath, $json + "`n", $utf8NoBom)
Write-Host "Native Windows Phase 4E validation passed."
Write-Host "Report: $reportPath"
