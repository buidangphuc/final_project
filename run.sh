#!/bin/bash

# =============================================================================
# One Pixel Attack - Reproduce Paper Results
# Paper: "One pixel attack for fooling deep neural networks"
# 
# Main Results Table:
# - CIFAR-10: AllConv, NiN, VGG16 (1, 3, 5 pixels)
# - ImageNet: BVLC AlexNet (1 pixel, non-targeted only)
# =============================================================================

set -e  # Exit on error

DEVICE="${DEVICE:-cuda}"  # Default to cuda, can override with: DEVICE=mps ./run.sh
TRAIN="${TRAIN:-1}"       # Set to 1 to train models, 0 to use existing checkpoints

echo "==============================================================================="
echo "ONE PIXEL ATTACK - REPRODUCING PAPER RESULTS"
echo "Device: $DEVICE | Train: $TRAIN"
echo "==============================================================================="

# -----------------------------------------------------------------------------
# PART 1: CIFAR-10 with All-CNN, NiN, VGG16
# Paper Table: Main results showing 1-pixel, 3-pixel, 5-pixel attacks
# -----------------------------------------------------------------------------

echo ""
echo "=== PART 1: CIFAR-10 - Three Networks (AllConv, NiN, VGG16) ==="
echo ""

# 1-PIXEL ATTACK (Main Result - all 3 networks)
# echo "[1/7] Running 1-pixel attack on AllConv, NiN, VGG16..."
# python exp.py \
#     --dataset cifar10 \
#     --models allconv nin vgg16 \
#     --pixels 1 \
#     --n_nontarget_samples 500 \
#     --population 400 \
#     --gen_max 100 \
#     --F 0.5 \
#     --earlystop_target_prob 0.9 \
#     --earlystop_trueprob 0.05 \
#     --run_targeted 0 \
#     --run_nontargeted 1 \
#     --device $DEVICE \
#     --train $TRAIN \
#     --epochs 100 \
#     --batch-size 128 \
#     --lr 0.1 \
#     --outdir results/cifar10_1pixel

# echo "✓ Completed 1-pixel attack (AllConv, NiN, VGG16)"
# echo ""

# # 3-PIXEL ATTACK (All 3 networks for comparison)
# echo "[2/7] Running 3-pixel attack on AllConv, NiN, VGG16..."
# python exp.py \
#     --dataset cifar10 \
#     --models allconv \
#     --pixels 3 \
#     --n_nontarget_samples 500 \
#     --population 400 \
#     --gen_max 100 \
#     --F 0.5 \
#     --earlystop_target_prob 0.9 \
#     --earlystop_trueprob 0.05 \
#     --run_targeted 1 \
#     --run_nontargeted 1 \
#     --device $DEVICE \
#     --train 0 \
#     --outdir results/cifar10_3pixel

# echo "✓ Completed 3-pixel attack (AllConv, NiN, VGG16)"
# echo ""

# 5-PIXEL ATTACK (All 3 networks for comparison)
# echo "[3/7] Running 5-pixel attack on AllConv, NiN, VGG16..."
# python exp.py \
#     --dataset cifar10 \
#     --models allconv \
#     --pixels 5 \
#     --n_nontarget_samples 500 \
#     --population 400 \
#     --gen_max 100 \
#     --F 0.5 \
#     --earlystop_target_prob 0.9 \
#     --earlystop_trueprob 0.05 \
#     --run_targeted 1 \
#     --run_nontargeted 1 \
#     --device $DEVICE \
#     --train 0 \
#     --outdir results/cifar10_5pixel

# echo "✓ Completed 5-pixel attack (AllConv, NiN, VGG16)"
# echo ""

# -----------------------------------------------------------------------------
# PART 2: ImageNet with BVLC AlexNet
# Paper Table: Non-targeted attack only on 105 ImageNet samples
# NOTE: Requires ImageNet data in data/imagenet_227/ (227x227 PNG images)
# -----------------------------------------------------------------------------

echo ""
echo "=== PART 2: ImageNet - AlexNet (Non-targeted only) ==="
echo ""

if [ -d "data/imagenet_227" ]; then
    echo "[4/7] Running 1-pixel non-targeted attack on AlexNet (ImageNet)..."
    python exp.py \
        --dataset imagenet_folder \
        --imagenet_folder data/imagenet_227 \
        --models alexnet \
        --pixels 1 \
        --n_nontarget_samples 105 \
        --run_targeted 0 \
        --run_nontargeted 1 \
        --population 400 \
        --gen_max 100 \
        --F 0.5 \
        --earlystop_trueprob 0.05 \
        --device $DEVICE \
        --outdir results/imagenet_1pixel
    
    echo "✓ Completed ImageNet AlexNet attack"
else
    echo "⚠ WARNING: ImageNet data not found at data/imagenet_227/"
    echo "  Skipping ImageNet experiments. To run ImageNet attacks:"
    echo "  1. Prepare 105 ImageNet validation images"
    echo "  2. Convert to PNG format (lossless)"
    echo "  3. Resize to 227x227 using nearest neighbor"
    echo "  4. Place in data/imagenet_227/"
fi
echo ""

# -----------------------------------------------------------------------------
# PART 3: Summary and Results Analysis
# -----------------------------------------------------------------------------

# echo ""
# echo "==============================================================================="
# echo "EXPERIMENTS COMPLETED"
# echo "==============================================================================="
# echo ""
# echo "Results saved in:"
# echo "  - results/cifar10_1pixel/    (1-pixel: AllConv, NiN, VGG16)"
# echo "  - results/cifar10_3pixel/    (3-pixel: AllConv, NiN, VGG16)"
# echo "  - results/cifar10_5pixel/    (5-pixel: AllConv, NiN, VGG16)"
# if [ -d "data/imagenet_227" ]; then
#     echo "  - results/imagenet_1pixel/   (1-pixel: AlexNet on ImageNet)"
# fi
# echo ""
# echo "Summary files:"
# echo "  - Each directory contains summary_results.json with success rates"
# echo "  - CSV files with per-image attack results"
# echo ""
# echo "To view results:"
# echo "  cat results/cifar10_1pixel/summary_results.json | python -m json.tool"
# echo ""
# echo "==============================================================================="