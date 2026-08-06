[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$NcnnSource = "",
    [string]$WorkRoot = "D:\hunyuanocr-recovery\phase4d",
    [ValidateRange(1, 64)]
    [int]$Jobs = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot "..\.."
}
$RepoRoot = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($RepoRoot))).FullName
if (-not $NcnnSource) {
    $NcnnSource = Join-Path (Split-Path $RepoRoot -Parent) "ncnn"
}
$NcnnSource = (Get-Item -LiteralPath ([IO.Path]::GetFullPath($NcnnSource))).FullName
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

$ncnnBuild = Join-Path $WorkRoot "ncnn-build"
$ncnnInstall = Join-Path $WorkRoot "ncnn-install"
$runtimeBuild = Join-Path $WorkRoot "runtime-build"
$runtimeInstall = Join-Path $WorkRoot "runtime-install"
$consumerBuild = Join-Path $WorkRoot "consumer-build"

$environment = @(
    "repo_root=$RepoRoot"
    "ncnn_source=$NcnnSource"
    "work_root=$WorkRoot"
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
    "-DHUNYUANOCR_BUILD_PARITY_TESTS=OFF"
)
Invoke-VsCommand -Log (Join-Path $WorkRoot "runtime_build_install.log") -Command (
    "cmake --build `"$runtimeBuild`" -j $Jobs && " +
    "cmake --install `"$runtimeBuild`""
)

$prefixPath = "$runtimeInstall;$ncnnInstall"
Invoke-VsCommand -Log (Join-Path $WorkRoot "consumer_find_package.log") -Command (
    "cmake -S `"$RepoRoot\tests\runtime_api_consumer`" " +
    "-B `"$consumerBuild`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
    "-DCMAKE_PREFIX_PATH=`"$prefixPath`" && " +
    "cmake --build `"$consumerBuild`" -j $Jobs && " +
    "`"$consumerBuild\runtime_api_consumer.exe`""
)

$cli = Join-Path $runtimeInstall "bin\hunyuanocr_cli.exe"
Invoke-VsCommand -Log (Join-Path $WorkRoot "windows_binary_dependencies.log") `
    -Command "dumpbin /headers `"$cli`" && dumpbin /dependents `"$cli`""

Write-Host "MSVC build, install, ncnn precision tests, and consumer passed."
Write-Host "Installed CLI: $cli"
