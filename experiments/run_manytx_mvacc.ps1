param(
    [int]$Seed = 7,
    [string]$SaveDir = "",
    [string]$DatasetPath = "C:\Users\123\Downloads\ManyTx.pkl\ManyTx.pkl",
    [int]$ReceiverIndex = 2,
    [int]$EqualizationIndex = 0,
    [int]$Epochs = 20,
    [switch]$TrainClosedSet,
    [switch]$DisableVisualization,
    [switch]$DisableRFAugmentation
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Experiment = Join-Path $PSScriptRoot "exp_manytx_multiview_graph_mvacc.py"

if ([string]::IsNullOrWhiteSpace($SaveDir)) {
    $SaveDir = ".\results\manytx_rx2_mvacc_trainingv2_seed$Seed"
}

$Arguments = @(
    $Experiment,
    "--dataset_path", $DatasetPath,
    "--save_dir", $SaveDir,
    "--selected_rx_list", "$ReceiverIndex",
    "--eq_index", "$EqualizationIndex",
    "--min_samples_per_tx", "40",
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
