param(
    [Parameter(Mandatory = $true)]
    [string]$InputH5ad,
    [Parameter(Mandatory = $true)]
    [string]$Genes,
    [Parameter(Mandatory = $true)]
    [string]$Perts,
    [Parameter(Mandatory = $true)]
    [string]$OutputVcc,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$vcc = Join-Path $env:USERPROFILE ".local\bin\vcc.exe"

$genesInput = Get-Content -LiteralPath $Genes
$genesFile = $Genes
$tempGenes = $null
if ($genesInput.Count -gt 0 -and $genesInput[0].Trim().ToLower() -eq "gene_name") {
    $tempGenes = Join-Path $env:TEMP ("vcc_genes_" + [guid]::NewGuid().ToString("N") + ".csv")
    $genesInput | Select-Object -Skip 1 | Set-Content -LiteralPath $tempGenes
    $genesFile = $tempGenes
}

try {
    $args = @("prep", $InputH5ad, "-g", $genesFile, "--perts", $Perts, "-o", $OutputVcc)
    if ($DryRun) {
        $args += "--dry-run"
    }
    & $vcc @args
}
finally {
    if ($tempGenes -and (Test-Path -LiteralPath $tempGenes)) {
        Remove-Item -LiteralPath $tempGenes -Force
    }
}
