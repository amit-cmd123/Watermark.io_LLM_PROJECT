param(
    [ValidateSet('test', 'smoke', 'experiment', 'ui', 'research')][string]$Action = 'test',
    [string]$PythonExe = '',
    [string]$Model = 'meta-llama/Llama-3.2-1B-Instruct',
    [int]$Limit = 20
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonExe) {
    $recorded = Join-Path $PSScriptRoot '.runtime\python-path.txt'
    if (Test-Path -LiteralPath $recorded) {
        $PythonExe = (Get-Content -LiteralPath $recorded -Raw).Trim()
    } elseif (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
        $PythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    } else {
        $PythonExe = 'python'
    }
}
$env:HF_HOME = Join-Path $PSScriptRoot '.runtime\huggingface'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.runtime\matplotlib'
if ($Action -eq 'test') {
    & $PythonExe -m unittest -v test_watermark
} elseif ($Action -eq 'smoke') {
    & $PythonExe smoke.py
} elseif ($Action -eq 'research') {
    & $PythonExe research_server.py
} elseif ($Action -eq 'ui') {
    & $PythonExe ui_server.py
} else {
    & $PythonExe evaluate.py --model $Model --limit $Limit
}
exit $LASTEXITCODE
