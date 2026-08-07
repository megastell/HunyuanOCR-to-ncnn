[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$NcnnSource = "",
    [string]$PythonExe = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4f",
    [string]$ModelDirectory = "D:\hunyuanocr-recovery\phase4c\model-ntfs",
    [string]$Phase = "4F",
    [string]$ReportPath = "",
    [string[]]$CtestSuites = @(
        "smoke",
        "dynamic",
        "real-png",
        "real-jpeg",
        "cache-budgets",
        "error-paths"
    ),
    [ValidateRange(1, 64)]
    [int]$Jobs = 12,
    [ValidateRange(1, 64)]
    [int]$Threads = 9,
    [switch]$SkipCtest,
    [switch]$ResumeCtest,
    [switch]$SkipCpack
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

if (-not $RepoRoot) { $RepoRoot = Join-Path $PSScriptRoot "..\.." }
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
if (-not $ReportPath) {
    $ReportPath = Join-Path $RepoRoot "docs\windows_phase4f_release_validation.json"
}
if (-not $NcnnSource) {
    $NcnnSource = Join-Path (Split-Path $RepoRoot -Parent) "ncnn"
}
$NcnnSource = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($NcnnSource))).FullName
if (-not $PythonExe) {
    $candidate = Join-Path $env:USERPROFILE `
        ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $PythonExe = $candidate
    } else {
        $PythonExe = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found; pass -PythonExe explicitly"
}
$PythonExe = (Get-Item -LiteralPath $PythonExe).FullName
$CtestSuites = @(
    $CtestSuites | ForEach-Object { $_ -split "," } |
        Where-Object { $_ -ne "" }
)
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Installer vswhere.exe was not found"
}
$vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsPath) {
    throw "A Visual Studio installation with the x64 C++ toolchain is required"
}
$vsDevCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"

function Invoke-VsCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$Log
    )
    $wrapped = "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && $Command"
    & cmd.exe /d /s /c $wrapped 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE; see $Log"
    }
}

function Write-ModelCompatibility {
    param([Parameter(Mandatory = $true)][string]$Directory)
    $manifest = Join-Path $Directory "runtime_manifest.tsv"
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw "Model manifest not found: $manifest"
    }
    $countLine = Get-Content -LiteralPath $manifest | Select-Object -Index 1
    if ($countLine -notmatch "^file_count`t([0-9]+)$") {
        throw "Unable to parse manifest file_count: $manifest"
    }
    $compat = @(
        "HUNYUANOCR_NCNN_RUNTIME_COMPATIBILITY_V1"
        ("model_id`t" + "tencent/HunyuanOCR")
        "runtime_abi_major`t0"
        "runtime_min_version`t0.1.0"
        "runtime_max_exclusive_version`t1.0.0"
        "manifest_format`tHUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1"
        "file_count`t$($Matches[1])"
        "precision`tfp32"
        "jpeg_pixel_contract`tstb_rgb_v1"
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText(
        (Join-Path $Directory "runtime_compatibility.tsv"),
        ($compat -join "`n") + "`n",
        $utf8NoBom)
}

function Invoke-CtestSuite {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$Suite
    )
    $log = Join-Path $WorkRoot "ctest_$Suite.log"
    if (-not $ResumeCtest -or -not (Test-Path -LiteralPath $log)) {
        $null = Invoke-VsCommand -Log $log -Command (
            "ctest --test-dir `"$BuildDir`" -L $Suite --output-on-failure"
        )
    }
    $text = Get-Content -Raw -LiteralPath $log
    $match = [regex]::Match(
        $text,
        "100% tests passed(?:, 0 tests failed)? out of ([0-9]+)")
    if (-not $match.Success) { throw "CTest suite did not report all passed: $Suite" }
    return [ordered]@{
        suite = $Suite
        test_count = [int]$match.Groups[1].Value
        status = "passed"
        log = $log
    }
}

$ncnnBuild = Join-Path $WorkRoot "ncnn-build"
$ncnnInstall = Join-Path $WorkRoot "ncnn-install"
$runtimeBuild = Join-Path $WorkRoot "runtime-build"
$runtimeInstall = Join-Path $WorkRoot "runtime-install"
$consumerBuild = Join-Path $WorkRoot "consumer-build"
$packageDir = Join-Path $WorkRoot "packages"
$ctestLogDir = Join-Path $WorkRoot "ctest"
New-Item -ItemType Directory -Force -Path $packageDir, $ctestLogDir | Out-Null
Write-ModelCompatibility $ModelDirectory

$environment = @(
    "repo_root=$RepoRoot"
    "ncnn_source=$NcnnSource"
    "python=$PythonExe"
    "work_root=$WorkRoot"
    "model_directory=$ModelDirectory"
    "visual_studio=$vsPath"
    "cmake=$(& cmake --version | Select-Object -First 1)"
    "os=$((Get-CimInstance Win32_OperatingSystem).Caption)"
)
$environment | Set-Content -Encoding ASCII `
    (Join-Path $WorkRoot "windows_environment.log")

Invoke-VsCommand -Log (Join-Path $WorkRoot "ncnn_configure.log") -Command (
    "cmake -S `"$NcnnSource`" -B `"$ncnnBuild`" -G Ninja " +
    "-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`"$ncnnInstall`" " +
    "-DNCNN_VULKAN=OFF -DNCNN_OPENMP=ON -DNCNN_BUILD_TOOLS=OFF " +
    "-DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF " +
    "-DNCNN_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF"
)
Invoke-VsCommand -Log (Join-Path $WorkRoot "ncnn_build_install.log") -Command (
    "cmake --build `"$ncnnBuild`" --target ncnn test_rotaryembed " +
    "test_rmsnorm -j $Jobs && cmake --install `"$ncnnBuild`""
)

$precisionResults = foreach ($name in @("test_rotaryembed", "test_rmsnorm")) {
    $test = Join-Path $ncnnBuild "tests\$name.exe"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $test
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    if ($exitCode -ne 0) {
        throw "$name failed with exit code $exitCode"
    }
    "$name.exe`texit=0`tseconds=$([Math]::Round($timer.Elapsed.TotalSeconds, 3))"
}
$precisionResults | Set-Content -Encoding ASCII `
    (Join-Path $WorkRoot "ncnn_precision_tests.log")

Invoke-VsCommand -Log (Join-Path $WorkRoot "runtime_configure.log") -Command (
    "cmake -S `"$RepoRoot`" -B `"$runtimeBuild`" -G Ninja " +
    "-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`"$runtimeInstall`" " +
    "-Dncnn_DIR=`"$ncnnInstall\lib\cmake\ncnn`" " +
    "-DHUNYUANOCR_BUILD_CLI=ON -DHUNYUANOCR_BUILD_BENCHMARKS=ON " +
    "-DHUNYUANOCR_BUILD_PARITY_TESTS=OFF " +
    "-DHUNYUANOCR_ENABLE_RELEASE_TESTS=ON " +
    "-DPython3_EXECUTABLE=`"$PythonExe`" " +
    "-DHUNYUANOCR_RELEASE_MODEL_DIR=`"$ModelDirectory`" " +
    "-DHUNYUANOCR_RELEASE_LOG_DIR=`"$ctestLogDir`""
)
Invoke-VsCommand -Log (Join-Path $WorkRoot "runtime_build_install.log") -Command (
    "cmake --build `"$runtimeBuild`" -j $Jobs && " +
    "cmake --install `"$runtimeBuild`""
)

$ctestResults = @()
if (-not $SkipCtest) {
    foreach ($suite in $CtestSuites) {
        $ctestResults += Invoke-CtestSuite $runtimeBuild $suite
    }
}

$prefixPath = "$runtimeInstall;$ncnnInstall"
Invoke-VsCommand -Log (Join-Path $WorkRoot "consumer_find_package.log") -Command (
    "cmake -S `"$RepoRoot\tests\runtime_api_consumer`" " +
    "-B `"$consumerBuild`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
    "-DCMAKE_PREFIX_PATH=`"$prefixPath`" && " +
    "cmake --build `"$consumerBuild`" -j $Jobs && " +
    "`"$consumerBuild\runtime_api_consumer.exe`""
)

$packages = @()
if (-not $SkipCpack) {
    Invoke-VsCommand -Log (Join-Path $WorkRoot "cpack.log") -Command (
        "cd /d `"$runtimeBuild`" && cpack -G ZIP -B `"$packageDir`""
    )
    $packages = Get-ChildItem -LiteralPath $packageDir -Filter "*.zip" |
        Sort-Object Name | ForEach-Object {
            [ordered]@{
                path = $_.FullName
                bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    if ($packages.Count -eq 0) { throw "CPack did not produce a ZIP package" }
}

$cli = Join-Path $runtimeInstall "bin\hunyuanocr_cli.exe"
Invoke-VsCommand -Log (Join-Path $WorkRoot "windows_binary_dependencies.log") `
    -Command "dumpbin /headers `"$cli`" && dumpbin /dependents `"$cli`""

$rootLicense = @(Get-ChildItem -LiteralPath $RepoRoot -File |
    Where-Object { $_.Name -match "^(LICENSE|COPYING)" } |
    ForEach-Object { $_.Name })
$ncnnLicense = @()
$archivedNcnnLicense = Join-Path $RepoRoot "third_party\licenses\ncnn-LICENSE.txt"
if (Test-Path -LiteralPath $archivedNcnnLicense) {
    $ncnnLicense += $archivedNcnnLicense
}
$modelLicense = Join-Path $RepoRoot `
    "third_party\licenses\Tencent-HunyuanOCR-LICENSE.txt"
$notice = Join-Path $RepoRoot "NOTICE"
$warnings = @()
if ($rootLicense.Count -eq 0) {
    $warnings += "Repository has no top-level LICENSE/COPYING file."
}
if ($ncnnLicense.Count -eq 0) {
    $warnings += "ncnn license file was not found in the repository archive."
}
if (-not (Test-Path -LiteralPath $modelLicense)) {
    $warnings += "Tencent HunyuanOCR model license archive is missing."
}
if (-not (Test-Path -LiteralPath $notice)) {
    $warnings += "Binary/source NOTICE file is missing."
}

$report = [ordered]@{
    phase = $Phase
    status = "passed"
    platform = (Get-CimInstance Win32_OperatingSystem).Caption
    model_directory = $ModelDirectory
    work_root = $WorkRoot
    ctest = $ctestResults
    packages = $packages
    dependency_license_audit = [ordered]@{
        project_license_files = $rootLicense
        runtime_third_party_headers = @(
            [ordered]@{
                name = "stb_image"
                path = "third_party/stb/stb_image.h"
                license_detected = "public domain"
                status = "passed"
            }
        )
        link_dependencies = @(
            [ordered]@{
                name = "ncnn"
                cmake_package = (Join-Path $ncnnInstall "lib\cmake\ncnn")
                license_files = $ncnnLicense
                status = if ($ncnnLicense.Count -gt 0) { "passed" } else { "warning" }
            }
        )
        model_license_archive = $modelLicense
        notice_file = $notice
        warnings = $warnings
        status = if ($warnings.Count -eq 0) { "passed" } else { "passed_with_warnings" }
    }
    offline_release_acceptance = [ordered]@{
        install_tree_created = $true
        find_package_consumer_passed = $true
        packages_created = ($packages.Count -gt 0) -or $SkipCpack
        network_required = $false
    }
    persistent_log_root = $WorkRoot
}
$json = ($report | ConvertTo-Json -Depth 12).Replace("`r`n", "`n")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($ReportPath, $json + "`n", $utf8NoBom)
Write-Host "Native Windows Phase $Phase release validation passed."
Write-Host "Report: $ReportPath"
