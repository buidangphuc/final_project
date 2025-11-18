import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from .attack import de_attack_image
    from .data_utils import ImagesFolderDataset, get_cifar_loaders
    from .models import AllConv, NiN, VGG16CIFAR, alexnet_bvlc
    from .training import evaluate_accuracy, train_model
    from .utils import set_seed
except ImportError:
    from attack import de_attack_image
    from data_utils import ImagesFolderDataset, get_cifar_loaders
    from models import AllConv, NiN, VGG16CIFAR, alexnet_bvlc
    from training import evaluate_accuracy, train_model
    from utils import set_seed


def _resolve_device(device_str: str) -> torch.device:
    # Normalize
    device_str = (device_str or "").lower()

    # Auto selection preference: CUDA -> MPS -> CPU
    if device_str in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Explicit CUDA
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        print("Requested CUDA but not available. Falling back to CPU.")

    # Explicit MPS (Apple Silicon)
    if device_str == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print("Requested MPS but not available. Falling back to CPU.")

    # CPU fallback
    return torch.device("cpu")


def _build_model(name: str, dataset: str, use_pretrained: bool = True) -> torch.nn.Module:
    """Build model with optional pretrained weights.
    
    Args:
        name: Model architecture name
        dataset: Dataset name (determines num_classes)
        use_pretrained: If True, load pretrained weights when available
    """
    name = name.lower()
    dataset = dataset.lower()
    num_classes = 1000 if dataset.startswith("imagenet") else 10
    
    if name == "allconv":
        return AllConv(num_classes=num_classes)
    if name == "nin":
        return NiN(num_classes=num_classes)
    if name == "vgg16":
        return VGG16CIFAR(num_classes=num_classes)
    if name == "alexnet":
        model = alexnet_bvlc(num_classes=num_classes, pretrained=use_pretrained)
        if use_pretrained and num_classes == 1000:
            print(f"Loaded pretrained AlexNet (ImageNet weights)")
        return model
    raise ValueError(f"Unknown model {name}")


def _prepare_datasets(args):
    dataset = args.dataset.lower()
    if dataset == "cifar10":
        train_loader, test_loader = get_cifar_loaders(
            batch_size=args.batch_size,
            workers=args.workers,
            no_augment=args.no_augment,
        )
        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "targeted_indices": _sample_indices(10000, args.n_targeted_samples, args.seed),
            "nontarget_indices": _sample_indices(10000, args.n_nontarget_samples, args.seed),
            "loader_fn": lambda idx: _load_cifar_item(test_loader.dataset, idx),
        }
    if dataset == "imagenet_folder":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_obj = ImagesFolderDataset(args.imagenet_folder, transform=transform)
        test_loader = DataLoader(dataset_obj, batch_size=1, shuffle=False)
        total = len(dataset_obj)
        return {
            "train_loader": None,
            "test_loader": test_loader,
            "targeted_indices": [],  # Paper doesn't do targeted attacks on ImageNet
            "nontarget_indices": _sample_indices(total, args.n_nontarget_samples, args.seed),
            "loader_fn": lambda idx: _load_folder_item(dataset_obj, idx),
        }
    raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'imagenet_folder'.")


def _sample_indices(total: int, count: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return indices[:count]


def _load_cifar_item(dataset, index: int) -> Tuple[np.ndarray, int, str]:
    tensor, label = dataset[index]
    img_np = tensor.permute(1, 2, 0).numpy()
    return img_np, int(label), f"cifar10_{index}"


def _load_folder_item(dataset, index: int) -> Tuple[np.ndarray, int, str]:
    tensor, label, fname = dataset[index]
    img_np = tensor.permute(1, 2, 0).numpy()
    parsed_label = label if label >= 0 else -1
    return img_np, int(parsed_label), fname


def _maybe_train_model(args, model, train_loader, device, ckpt_path: Path) -> torch.nn.Module:
    """Load checkpoint if available, otherwise train if requested."""
    if ckpt_path.exists():
        print(f"Loading trained checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        return model
    
    if args.train and train_loader is not None:
        print("Training model (this may take long)...")
        return train_model(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            ckpt_path=str(ckpt_path),
        )
    
    print("No checkpoint found and training disabled. Using model as-is (may have pretrained weights).")
    return model


def run_experiments(args) -> None:
    device = _resolve_device(args.device)
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = _prepare_datasets(args)
    train_loader = dataset_bundle["train_loader"]
    test_loader = dataset_bundle["test_loader"]
    targeted_indices = dataset_bundle["targeted_indices"]
    nontarget_indices = dataset_bundle["nontarget_indices"]
    load_item = dataset_bundle["loader_fn"]

    summary: Dict[str, Dict[str, object]] = {}

    for model_name in args.models:
        print(f"=== Model {model_name} ===")
        use_pretrained = getattr(args, 'use_pretrained', True)
        model = _build_model(model_name, args.dataset, use_pretrained=use_pretrained)
        ckpt_dir = Path("models")
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"{model_name}_{args.dataset}.pth"
        model = _maybe_train_model(args, model, train_loader, device, ckpt_path)

        if args.dataset.lower() == "cifar10" and test_loader is not None:
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"Clean accuracy on CIFAR-10 test: {acc * 100:.2f}%")
        else:
            acc = None

        model = model.to(device)
        model.eval()

        model_summary = {"clean_acc": acc, "results": {}}

        if args.run_targeted and targeted_indices:
            print(f"Running targeted attacks on {len(targeted_indices)} images, trying {args.num_targets_each} targets each.")
            targeted_rows = []
            for idx_count, index in enumerate(targeted_indices, start=1):
                img_np, true_lbl, fname = load_item(index)
                if true_lbl < 0:
                    print(f"Skipping {fname} because label unknown")
                    continue
                targets = [c for c in range(10) if c != true_lbl][: args.num_targets_each]
                for target in targets:
                    result = de_attack_image(
                        model,
                        device,
                        img_np,
                        true_label=true_lbl,
                        target_label=target,
                        pixels=args.pixels,
                        population=args.population,
                        differential_weight=args.F,
                        max_generations=args.gen_max,
                        mode="targeted",
                        earlystop_target_prob=args.earlystop_target_prob,
                    )
                    targeted_rows.append(
                        {
                            "image_index": index,
                            "fname": fname,
                            "true_label": true_lbl,
                            "target_label": target,
                            "success": result.get("success", False),
                            "n_evals": result.get("n_evals"),
                            "best_f": result.get("best_f"),
                        }
                    )
                if idx_count % 50 == 0:
                    print(f"Processed {idx_count}/{len(targeted_indices)} targeted images")
            csv_path = outdir / f"{model_name}_{args.dataset}_targeted_pixels{args.pixels}.csv"
            pd.DataFrame(targeted_rows).to_csv(csv_path, index=False)
            success_count = sum(1 for row in targeted_rows if row["success"])
            total = len(targeted_rows)
            rate = 100 * success_count / total if total else 0.0
            print(f"Targeted: successes {success_count}/{total} rate {rate:.2f}%")
            model_summary["results"]["targeted"] = {"csv": str(csv_path), "success": success_count, "total": total}

        if args.run_nontargeted and nontarget_indices:
            print(f"Running non-targeted attacks on {len(nontarget_indices)} images.")
            nontarget_rows = []
            for idx_count, index in enumerate(nontarget_indices, start=1):
                img_np, true_lbl, fname = load_item(index)
                if true_lbl < 0:
                    print(f"Skipping {fname} because label unknown")
                    continue
                result = de_attack_image(
                    model,
                    device,
                    img_np,
                    true_label=true_lbl,
                    target_label=None,
                    pixels=args.pixels,
                    population=args.population,
                    differential_weight=args.F,
                    max_generations=args.gen_max,
                    mode="nontarget",
                    earlystop_trueprob=args.earlystop_trueprob,
                )
                nontarget_rows.append(
                    {
                        "image_index": index,
                        "fname": fname,
                        "true_label": true_lbl,
                        "success": result.get("success", False),
                        "n_evals": result.get("n_evals"),
                        "best_f": result.get("best_f"),
                        "final_pred": result.get("final_pred"),
                    }
                )
                if idx_count % 100 == 0:
                    print(f"Processed {idx_count}/{len(nontarget_indices)} non-target images")
            csv_path = outdir / f"{model_name}_{args.dataset}_nontargeted_pixels{args.pixels}.csv"
            pd.DataFrame(nontarget_rows).to_csv(csv_path, index=False)
            success_count = sum(1 for row in nontarget_rows if row["success"])
            total = len(nontarget_rows)
            rate = 100 * success_count / total if total else 0.0
            print(f"Non-targeted: successes {success_count}/{total} rate {rate:.2f}%")
            model_summary["results"]["nontargeted"] = {"csv": str(csv_path), "success": success_count, "total": total}

        summary[model_name] = model_summary
        with open(outdir / "summary_results.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

    print("All experiments finished. Summary saved to", str(outdir / "summary_results.json"))
