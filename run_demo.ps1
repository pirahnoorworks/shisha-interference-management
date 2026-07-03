$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\demo_interference.py --jobs 300 --seed 42 --plot
