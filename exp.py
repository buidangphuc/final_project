#!/usr/bin/env python3

import argparse

try:
    from .experiment import run_experiments
except ImportError:  # pragma: no cover - allow running as script
    from experiment import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet_folder"])
    parser.add_argument("--imagenet_folder", type=str, default="data/imagenet_227", help="folder with imagenet PNGs 227x227")
    parser.add_argument("--models", nargs="+", default=None, help="models to run (auto-select based on dataset if not specified)")
    parser.add_argument("--train", type=int, default=0, help="1 to train models if checkpoint missing")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="onepixel_results")
    parser.add_argument("--pixels", type=int, default=1, choices=[1, 3, 5])
    parser.add_argument("--population", type=int, default=400, help="DE population size (paper uses 400)")
    parser.add_argument("--F", type=float, default=0.5, help="DE differential weight (paper uses 0.5)")
    parser.add_argument("--gen_max", type=int, default=100, help="Max DE generations")
    parser.add_argument("--earlystop_target_prob", type=float, default=0.9, help="Stop targeted attack if target class prob >= this")
    parser.add_argument("--earlystop_trueprob", type=float, default=0.05, help="Stop non-targeted if true class prob <= this")
    parser.add_argument("--n_targeted_samples", type=int, default=100, help="Number of images for targeted attack (paper uses subset)")
    parser.add_argument("--n_nontarget_samples", type=int, default=500, help="Number of images for non-targeted (paper: 500 for CIFAR, 105 for ImageNet)")
    parser.add_argument("--num_targets_each", type=int, default=9, help="Number of target classes to try per image (paper uses 9)")
    parser.add_argument("--run_targeted", type=int, default=1, help="1 to run targeted attacks")
    parser.add_argument("--run_nontargeted", type=int, default=1, help="1 to run non-targeted attacks")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    
    # Auto-select models based on dataset (as per paper)
    if arguments.models is None:
        if arguments.dataset == "cifar10":
            arguments.models = ["allconv", "nin", "vgg16"]  # Paper Table II
            print("Using CIFAR-10 models: All-CNN, NiN, VGG16")
        elif arguments.dataset == "imagenet_folder":
            arguments.models = ["alexnet"]  # Paper Table III
            arguments.n_nontarget_samples = 105  # Paper uses 105 ImageNet samples
            arguments.run_targeted = 0  # Paper only does non-targeted for ImageNet
            print("Using ImageNet model: AlexNet (105 samples, non-targeted only)")
    
    run_experiments(arguments)
