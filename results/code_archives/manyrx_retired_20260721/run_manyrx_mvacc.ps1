param(
    [ValidateSet("fixedrx", "crossrx")]
    [string]$Protocol = "fixedrx",
    [int]$Seed = 7,
    [string]$SaveDir = "",
    [int]$Epochs = 20,
    [switch]$TrainClosedSet,
    [switch]$DisableVisualization,
    [switch]$DisableRFAugmentation
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Experiment = Join-Path $PSScriptRoot "exp_manyrx_multiview_graph_mvacc.py"
$Dataset = Join-Path $ProjectRoot "datasets\manyrx_compact\manyrx_4known_3round_fixed_cross_rx.npz"
$Manifest = Join-Path $ProjectRoot "datasets\manyrx_compact\manyrx_protocol_manifest.json"

if ([string]::IsNullOrWhiteSpace($SaveDir)) {
    $SaveDir = ".\results\manyrx_${Protocol}_trainingv2_seed$Seed"
}

$Arguments = @(
    $Experiment,
    "--dataset_path", $Dataset,
    "--source_manifest", $Manifest,
    "--protocol", $Protocol,
    "--save_dir", $SaveDir,
    "--epochs", "$Epochs",
    "--batch_size", "128",
    "--test_batch_size", "256",
    "--lr", "0.001",
    "--feat_dim", "128",
    "--use_supcon",
    "--supcon_weight", "0.1",
    "--supcon_temperature", "0.2",
    "--closedset_val_ratio", "0.142857142857",
    "--projection_hidden_dim", "128",
    "--projection_dim", "64",
    "--adaptive_fusion",
    "--visualization_method", "both",
    "--disable_cil_baselines",
    "--seed", "$Seed"
)

if ($TrainClosedSet) {
    $Arguments += "--train_closedset"
}
if ($DisableVisualization) {
    $Arguments += "--disable_visualization"
}
if ($DisableRFAugmentation) {
    $Arguments += "--disable_rf_augmentation"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
