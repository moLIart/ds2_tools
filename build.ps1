param(
    [string]$QtRoot = 'C:/Qt/6.11.0/msvc2022_64',
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')][string]$Configuration = 'Release',
    [string]$BuildDirectory = "$PSScriptRoot/build"
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    if (Get-Command vsenv -ErrorAction SilentlyContinue) { vsenv }
    else {
        $vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
        $vs = & $vswhere -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw 'MSVC x64 tools not found. Activate vsenv first.' }
        $vcvars = "$vs/VC/Auxiliary/Build/vcvarsall.bat"
        $env:PATH += ";$(Split-Path $vswhere)"
        cmd /c "`"$vcvars`" x64 && set" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
        }
        if ($LASTEXITCODE) { throw 'MSVC environment activation failed.' }
    }
}
$cmake = 'C:/Qt/Tools/CMake_64/bin/cmake.exe'
$ninja = 'C:/Qt/Tools/Ninja/ninja.exe'
if (-not (Test-Path $cmake)) { $cmake = (Get-Command cmake -ErrorAction Stop).Source }
if (-not (Test-Path $ninja)) { $ninja = (Get-Command ninja -ErrorAction Stop).Source }
& $cmake -S $PSScriptRoot -B $BuildDirectory -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_PREFIX_PATH=$QtRoot" "-DCMAKE_BUILD_TYPE=$Configuration"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $cmake --build $BuildDirectory
exit $LASTEXITCODE
