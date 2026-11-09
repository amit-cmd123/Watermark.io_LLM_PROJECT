param(
    [string]$Model = 'Qwen/Qwen2.5-0.5B-Instruct',
    [string]$Revision = 'main',
    [string]$PythonExe = ''
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
$cachePath = Join-Path $PSScriptRoot '.runtime\huggingface\hub'
Write-Host "Downloading $Model for readable paragraph generation. This may take several minutes."
$downloadCode = @'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3],
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
print('Model ready. Return to Watermark Lab and click Generate two samples.')
'@
& $PythonExe -c $downloadCode $Model $Revision $cachePath
if ($LASTEXITCODE -ne 0) {
    throw 'Model download failed. Check internet access and the error above, then run this script again to resume.'
}
