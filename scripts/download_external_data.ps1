# Download public perturbation data used for local evaluation.
#
# The 2026 Challenge releases no training set and no ground truth, so the only way to
# score a model locally is against an external dataset that has measured perturbation
# effects. Replogle et al. 2022 fits: it covers two different cell lines (K562 and RPE1)
# with largely overlapping CRISPRi targets, which makes it a local analogue of the
# Challenge's cross-cell-line generalization task.
#
# Files come from the scPerturb harmonized release (Zenodo record 13350497).
#
#   .\scripts\download_external_data.ps1              # K562 essential + RPE1 (~2.8 GB)
#   .\scripts\download_external_data.ps1 -IncludeGwps # also the genome-wide K562 (~8.8 GB)

param(
    [string]$OutDir = "external",
    [switch]$IncludeGwps,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$record = "https://zenodo.org/api/records/13350497/files"

$files = @(
    @{ Name = "ReplogleWeissman2022_K562_essential.h5ad"; SizeGb = 1.54 }
    @{ Name = "ReplogleWeissman2022_rpe1.h5ad";           SizeGb = 1.24 }
)
if ($IncludeGwps) {
    $files += @{ Name = "ReplogleWeissman2022_K562_gwps.h5ad"; SizeGb = 8.81 }
}

if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

foreach ($file in $files) {
    $target = Join-Path $OutDir $file.Name

    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        $haveGb = [math]::Round((Get-Item -LiteralPath $target).Length / 1GB, 2)
        Write-Host "skip  $($file.Name) (already present, $haveGb GB)"
        continue
    }

    $url = "$record/$($file.Name)/content"
    $partial = "$target.partial"

    Write-Host "fetch $($file.Name) (~$($file.SizeGb) GB)"
    $started = Get-Date

    # Invoke-WebRequest buffers the whole body in memory; BITS streams to disk instead.
    Start-BitsTransfer -Source $url -Destination $partial -Description $file.Name

    Move-Item -LiteralPath $partial -Destination $target -Force
    $elapsed = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    $gotGb = [math]::Round((Get-Item -LiteralPath $target).Length / 1GB, 2)
    Write-Host "done  $($file.Name) ($gotGb GB in $elapsed min)"
}

Write-Host ""
Write-Host "Files in ${OutDir}:"
Get-ChildItem -LiteralPath $OutDir | Select-Object Name, @{N = "GB"; E = { [math]::Round($_.Length / 1GB, 2) } }
