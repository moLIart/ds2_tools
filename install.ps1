param(
    [string]$BuildDirectory = "$PSScriptRoot/build",
    [string]$Prefix = "$PSScriptRoot/install"
)
$ErrorActionPreference = 'Stop'
$cmake = 'C:/Qt/Tools/CMake_64/bin/cmake.exe'
if (-not (Test-Path $cmake)) { $cmake = (Get-Command cmake -ErrorAction Stop).Source }
& $cmake --install $BuildDirectory --config Release --prefix $Prefix
exit $LASTEXITCODE
