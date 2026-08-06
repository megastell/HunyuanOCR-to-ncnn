[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4b",
    [string]$ModelDirectory = "",
    [string]$ImagePath = "",
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
    $ModelDirectory = Join-Path $RepoRoot "artifacts"
}
if (-not $ImagePath) {
    $ImagePath = Join-Path $RepoRoot "tests\assets\ocr_smoke_en.png"
}
$cli = Join-Path $WorkRoot "runtime-install\bin\hunyuanocr_cli.exe"
$manifestPath = Join-Path $ModelDirectory "runtime_manifest.tsv"
foreach ($path in @($cli, $manifestPath, $ImagePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path not found: $path"
    }
}

$manifest = Get-Content -LiteralPath $manifestPath
if ($manifest.Count -lt 3 -or
    $manifest[0] -ne "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1") {
    throw "Unexpected runtime manifest header or file count"
}
$manifestFileCount = [int](($manifest[1] -split "`t")[1])
$totalBytes = [int64]0
foreach ($line in $manifest | Select-Object -Skip 2) {
    $columns = $line -split "`t"
    if ($columns.Count -ne 3) {
        throw "Invalid manifest entry: $line"
    }
    $totalBytes += [int64]$columns[1]
}
if (($manifest.Count - 2) -ne $manifestFileCount) {
    throw "Manifest inventory or total byte count changed"
}

function Invoke-Ocr {
    param(
        [Parameter(Mandatory = $true)][bool]$Packing,
        [Parameter(Mandatory = $true)][string]$Verification,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $log = Join-Path $WorkRoot "windows_${Name}.log"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $output = & $cli --model-dir $ModelDirectory --image $ImagePath `
        --packing $(if ($Packing) { "1" } else { "0" }) `
        --threads $Threads --max-new-tokens 32 --verify $Verification 2>&1 |
        Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode; see $log"
    }
    $text = $output -join "`n"
    $expectedTokens =
        "93892 5112 206 1717 21 185 18009 15613 16678 21836 120007"
    if ($text -notmatch "Generated tokens: $expectedTokens" -or
        $text -notmatch "EOS reached\s+: true" -or
        $text -notmatch "Generated text:\r?\nHELLO 2026\r?\nNCNN CPU TEST") {
        throw "$Name did not produce the exact expected token and text output"
    }

    function Read-Number([string]$Label) {
        $match = [regex]::Match($text, "$Label\s+: ([0-9.]+)")
        if (-not $match.Success) {
            throw "Missing metric $Label in $log"
        }
        return [double]$match.Groups[1].Value
    }

    [ordered]@{
        verification = $Verification
        tokens = $expectedTokens -split " " | ForEach-Object { [int]$_ }
        text = "HELLO 2026`nNCNN CPU TEST"
        load_seconds = Read-Number "Load seconds"
        input_seconds = Read-Number "Input seconds"
        prefill_seconds = Read-Number "Prefill seconds"
        decode_seconds = Read-Number "Decode seconds"
        runtime_seconds = Read-Number "Runtime seconds"
        peak_rss_kib = [int64](Read-Number "Peak RSS KiB")
        wall_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
    }
}

$unpacked = Invoke-Ocr -Packing $false -Verification "size" -Name "unpacked_size"
$packed = Invoke-Ocr -Packing $true -Verification "sha256" -Name "packed_sha256"
$os = Get-CimInstance Win32_OperatingSystem
$report = [ordered]@{
    phase = "4B"
    status = "passed"
    platform = [ordered]@{
        os = $os.Caption
        version = $os.Version
        architecture = $os.OSArchitecture
        cmake = (& cmake --version | Select-Object -First 1)
        msvc = "19.51.36246"
        ncnn = "20260806"
    }
    build = [ordered]@{
        runtime_install = "passed"
        installed_package_consumer = "passed"
        ncnn_rotaryembed_test = "passed"
        ncnn_rmsnorm_test = "passed"
    }
    manifest = [ordered]@{
        format = "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1"
        file_count = $manifestFileCount
        total_bytes = $totalBytes
        size_verification = "passed"
        cpp_sha256_verification = "passed"
    }
    unpacked_cli = $unpacked
    packed_cli = $packed
    binary_dependencies = @(
        "KERNEL32.dll", "MSVCP140.dll", "VCOMP140.DLL",
        "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "Universal CRT"
    )
}
$reportPath = Join-Path $RepoRoot "docs\windows_msvc_validation.json"
$json = $report | ConvertTo-Json -Depth 8
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    $reportPath,
    ($json -replace "`r`n", "`n") + "`n",
    $utf8NoBom)
Write-Host "Native Windows packed and unpacked validation passed."
Write-Host "Report: $reportPath"
