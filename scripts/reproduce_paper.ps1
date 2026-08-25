param(
    [string]$OutputRoot = "results/reproduce_paper",
    [switch]$IncludeCifarPilot
)

$ErrorActionPreference = "Stop"
$canonicalOutput = Join-Path $OutputRoot "canonical"
New-Item -ItemType Directory -Force -Path $canonicalOutput | Out-Null

python scripts/benchmark_new_version.py `
    --output (Join-Path $canonicalOutput "new_version_image_comparison_5seed.json") `
    --seeds 7 17 27 37 47 `
    --num-tasks 4 --classes-per-task 3 `
    --train-samples-per-class 24 --test-samples-per-class 12 `
    --image-size 32 --noise 0.1 --epochs 5 --batch-size 32

python scripts/benchmark_current_baselines.py `
    --output (Join-Path $canonicalOutput "current_backbone_baselines.json") `
    --seeds 7 17 27 37 47 `
    --num-tasks 4 --classes-per-task 3 `
    --train-samples-per-class 24 --test-samples-per-class 12 `
    --image-size 32 --noise 0.1 --epochs 5 --batch-size 32

if ($IncludeCifarPilot) {
    python scripts/benchmark_real_replay.py `
        --output (Join-Path $OutputRoot "real_replay_cifar100_review.json") `
        --data-root data/cifar100 --seeds 7 17 27 `
        --order-count 3 --epochs 2 `
        --train-samples-per-class 8 --test-samples-per-class 20 `
        --memory-per-task 20 --batch-size 64
}

Write-Output "Reproduction outputs written to $OutputRoot"
