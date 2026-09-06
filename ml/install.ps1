$ErrorActionPreference = 'Stop'

$python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
if ($null -eq $python) { throw 'Python 3 was not found. Install Python 3.10+ and rerun this script.' }

$venv = Join-Path $PSScriptRoot '.venv'
& $python.Source -m venv $venv
$venvPython = Join-Path $venv 'Scripts\python.exe'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
Write-Host "ML dependencies installed in $venv"
Write-Host "Run: .\ml\.venv\Scripts\python.exe .\ml\export_features.py"
