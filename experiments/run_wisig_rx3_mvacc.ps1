param(
    [int]$Seed = 7,
    [string]$SaveDir = "",
    [string]$Checkpoint = "",
    [switch]$TrainClosedSet,
    [switch]$DisableVisualization,
    [switch]$DisableRFAugmentation
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Experiment = Join-Path $PSScriptRoot "exp_wisig_multiview_graph_hard_end_to_end_comparison_MVACC.py"

if ([string]::IsNullOrWhiteSpace($SaveDir)) {
    $SaveDir = ".\results\wisig_rx3_mvacc_cflcg_v3_seed$Seed"
}

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $SaveDir "closedset_rx3_day1_10known_cflcg_v3.pth"
}

$Arguments = @(
    $Experiment,
    "--dataset_path", "D:\WiSigCustom\WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl",
    "--save_dir", $SaveDir,
    "--checkpoint", $Checkpoint,
    "--selected_rx_list", "2",
    "--epochs", "20",
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
    "--disable_cil_baselines",
    "--seed", "$Seed"
)

if ($TrainClosedSet) {
    $Arguments += "--train_closedset"
}
if ($DisableRFAugmentation) {
    $Arguments += "--disable_rf_augmentation"
}
if ($DisableVisualization) {
    $Arguments += "--disable_visualization"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
