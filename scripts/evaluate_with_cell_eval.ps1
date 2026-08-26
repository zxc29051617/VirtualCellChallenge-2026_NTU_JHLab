$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory = $true)]
    [string]$Prediction,
    [Parameter(Mandatory = $true)]
    [string]$Reference,
    [string]$OutDir = "artifacts/cell_eval"
)

uv tool run cell-eval run -ap $Prediction -ar $Reference --profile vcc --allow-discrete -o $OutDir

