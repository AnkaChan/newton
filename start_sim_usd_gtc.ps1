$ErrorActionPreference = "Stop"

# Run from repository root
Set-Location -Path $PSScriptRoot

$usdPath = "D:\Data\GTC2025DC_Demo\Inputs\SceneB\1018\20251018_to_sim_inSimClothB_01_physics.usd"
$scriptPath = Join-Path $PSScriptRoot "newton\_src\utils\sim_usd_gtc.py"

# Use conda without needing to activate the shell environment
conda run -n warp-build python "$scriptPath" "$usdPath" -n 1200 -i vbd




