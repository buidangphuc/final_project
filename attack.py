import time
from typing import Any, Dict, Optional

import numpy as np
import torch


def clip_rgb(candidates: np.ndarray) -> np.ndarray:
    candidates[..., 2:5] = np.clip(candidates[..., 2:5], 0.0, 255.0)
    return candidates


def init_population(npop: int, pixels: int, width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    population = np.zeros((npop, pixels, 5), dtype=np.float32)
    population[..., 0] = rng.uniform(1, width + 1, size=(npop, pixels))
    population[..., 1] = rng.uniform(1, height + 1, size=(npop, pixels))
    population[..., 2:5] = rng.normal(loc=128.0, scale=127.0, size=(npop, pixels, 3))
    return clip_rgb(population)


def apply_candidate_to_image(img: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    output = img.copy().astype(np.float32)
    if output.max() <= 1.0 + 1e-9:
        output *= 255.0
    height, width, _ = output.shape
    for px in candidate:
        x = int(round(px[0])) - 1
        y = int(round(px[1])) - 1
        if 0 <= x < width and 0 <= y < height:
            output[y, x, 0:3] = px[2:5]
    output = np.clip(output, 0.0, 255.0).astype(np.float32) / 255.0
    return output


def de_attack_image(
    model: torch.nn.Module,
    device: torch.device,
    img_np_orig: np.ndarray,
    true_label: int,
    target_label: Optional[int] = None,
    pixels: int = 1,
    population: int = 400,
    differential_weight: float = 0.5,
    max_generations: int = 100,
    earlystop_target_prob: float = 0.9,
    earlystop_trueprob: float = 0.05,
    mode: str = "targeted",
) -> Dict[str, Any]:
    model.eval()
    height, width, _ = img_np_orig.shape
    rng = np.random.RandomState(int(time.time()) % 2 ** 32)
    pop = init_population(population, pixels, width, height, seed=rng.randint(0, 2 ** 31 - 1))

    def eval_candidates(candidates_array: np.ndarray) -> np.ndarray:
        imgs = [apply_candidate_to_image(img_np_orig, cand) for cand in candidates_array]
        stacked = np.stack(imgs, axis=0)
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        if mode == "targeted" and target_label is not None:
            return probs[:, target_label]
        return -probs[:, true_label]

    fitness = eval_candidates(pop)
    evaluations = pop.shape[0]
    best_idx = int(fitness.argmax())
    best_candidate = pop[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    if mode == "targeted" and best_fitness >= earlystop_target_prob:
        return {"success": True, "candidate": best_candidate, "n_evals": evaluations, "best_f": best_fitness}
    if mode == "nontarget" and best_fitness <= -earlystop_trueprob:
        return {"success": True, "candidate": best_candidate, "n_evals": evaluations, "best_f": best_fitness}

    for _ in range(max_generations):
        for idx in range(population):
            choices = list(range(population))
            choices.remove(idx)
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = pop[r1] + differential_weight * (pop[r2] - pop[r3])
            mutant[..., 0] = np.clip(mutant[..., 0], 1.0, float(width))
            mutant[..., 1] = np.clip(mutant[..., 1], 1.0, float(height))
            mutant[..., 2:5] = np.clip(mutant[..., 2:5], 0.0, 255.0)
            mutant_fitness = eval_candidates(mutant[np.newaxis, ...])[0]
            evaluations += 1
            if mutant_fitness > fitness[idx]:
                pop[idx] = mutant
                fitness[idx] = mutant_fitness
                if mutant_fitness > best_fitness:
                    best_candidate = mutant.copy()
                    best_fitness = float(mutant_fitness)
                    if mode == "targeted" and best_fitness >= earlystop_target_prob:
                        return {
                            "success": True,
                            "candidate": best_candidate,
                            "n_evals": evaluations,
                            "best_f": best_fitness,
                        }
                    if mode == "nontarget" and best_fitness <= -earlystop_trueprob:
                        return {
                            "success": True,
                            "candidate": best_candidate,
                            "n_evals": evaluations,
                            "best_f": best_fitness,
                        }
    perturbed = apply_candidate_to_image(img_np_orig, best_candidate)
    tensor = torch.from_numpy(perturbed).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    prediction = int(probs.argmax())
    success = prediction == target_label if mode == "targeted" and target_label is not None else prediction != true_label
    return {
        "success": bool(success),
        "candidate": best_candidate,
        "n_evals": evaluations,
        "best_f": best_fitness,
        "final_probs": probs.tolist(),
        "final_pred": prediction,
    }
