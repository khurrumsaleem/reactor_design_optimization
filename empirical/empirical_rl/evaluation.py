from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from .model import ReactionPolicy
from .training import measured_reward


def evaluate_uniform(
    catalog, targets: Sequence[float], objective: str, restricted: bool
) -> list[dict[str, float | int]]:
    """Exact equal-probability search baseline over measured catalog rows."""
    metrics: list[dict[str, float | int]] = []
    for aryl in range(catalog.n_aryls):
        for target in targets:
            rows = [catalog.reactions[i] for i in catalog.rows_by_aryl[aryl]]
            if restricted:
                rows = [row for row in rows if catalog.is_in_training_support(row)]
            probability = 1.0 / len(rows)
            yields = np.asarray([row.measured_yield for row in rows], dtype=float)
            rewards = np.asarray(
                [measured_reward(value, float(target), objective) for value in yields]
            )
            expanded = np.asarray(
                [not catalog.is_in_training_support(row) for row in rows], dtype=float
            )
            ligand_probs = np.zeros(catalog.n_ligands, dtype=float)
            for row in rows:
                ligand_probs[row.ligand] += probability
            positive = ligand_probs[ligand_probs > 0]
            restricted_errors = [
                abs(row.measured_yield - target)
                for row in catalog.reactions
                if row.aryl == aryl and catalog.is_in_training_support(row)
            ]
            full_errors = [
                abs(row.measured_yield - target)
                for row in catalog.reactions
                if row.aryl == aryl
            ]
            metrics.append(
                {
                    "aryl_id": aryl,
                    "target": float(target),
                    "expected_reward": float(rewards.mean()),
                    "expected_abs_error": float(np.abs(yields - target).mean()),
                    "expansion_mass": float(expanded.mean()),
                    "ligand_entropy": float(-(positive * np.log(positive)).sum()),
                    "mode_row_id": rows[0].row_id,
                    "mode_yield": rows[0].measured_yield,
                    "full_oracle_abs_error": float(min(full_errors)),
                    "restricted_oracle_abs_error": float(min(restricted_errors)),
                }
            )
    return metrics


@torch.no_grad()
def evaluate_policy(
    policy: ReactionPolicy, targets: Sequence[float], objective: str
) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    for aryl in range(policy.catalog.n_aryls):
        for target_index, target in enumerate(targets, start=1):
            row_ids, probs_t = policy.exact_distribution(aryl, target_index)
            probs = probs_t.cpu().numpy()
            rows = [policy.catalog.reactions[i] for i in row_ids]
            yields = np.asarray([r.measured_yield for r in rows], dtype=float)
            rewards = np.asarray(
                [measured_reward(value, float(target), objective) for value in yields]
            )
            expanded = np.asarray(
                [not policy.catalog.is_in_training_support(r) for r in rows], dtype=float
            )
            ligand_probs = np.zeros(policy.catalog.n_ligands, dtype=float)
            for row, probability in zip(rows, probs):
                ligand_probs[row.ligand] += probability
            positive = ligand_probs[ligand_probs > 0]
            ligand_entropy = float(-(positive * np.log(positive)).sum())
            restricted_errors = [
                abs(r.measured_yield - target)
                for r in rows
                if policy.catalog.is_in_training_support(r)
            ]
            best_index = int(np.argmax(probs))
            metrics.append(
                {
                    "aryl_id": aryl,
                    "target": float(target),
                    "expected_reward": float(np.dot(probs, rewards)),
                    "expected_abs_error": float(np.dot(probs, np.abs(yields - target))),
                    "expansion_mass": float(np.dot(probs, expanded)),
                    "ligand_entropy": ligand_entropy,
                    "mode_row_id": row_ids[best_index],
                    "mode_yield": float(yields[best_index]),
                    "full_oracle_abs_error": float(np.min(np.abs(yields - target))),
                    "restricted_oracle_abs_error": float(min(restricted_errors)),
                }
            )
    return metrics


@torch.no_grad()
def policy_jsd(first: ReactionPolicy, second: ReactionPolicy, targets: Sequence[float]) -> float:
    values: list[float] = []
    for aryl in range(first.catalog.n_aryls):
        for target_index, _ in enumerate(targets, start=1):
            ids_a, pa_t = first.exact_distribution(aryl, target_index)
            ids_b, pb_t = second.exact_distribution(aryl, target_index)
            if ids_a != ids_b:
                raise ValueError("policies do not share the same empirical catalog")
            pa = pa_t.cpu().numpy()
            pb = pb_t.cpu().numpy()
            midpoint = 0.5 * (pa + pb)
            kl_a = np.sum(np.where(pa > 0, pa * np.log(pa / midpoint), 0.0))
            kl_b = np.sum(np.where(pb > 0, pb * np.log(pb / midpoint), 0.0))
            values.append(float(0.5 * (kl_a + kl_b)))
    return float(np.mean(values))


@torch.no_grad()
def algorithm_ligand_nmi(
    first: ReactionPolicy, second: ReactionPolicy, targets: Sequence[float]
) -> float:
    joint = np.zeros((2, first.catalog.n_ligands), dtype=float)
    condition_count = first.catalog.n_aryls * len(targets)
    for algorithm, policy in enumerate((first, second)):
        for aryl in range(policy.catalog.n_aryls):
            for target_index, _ in enumerate(targets, start=1):
                row_ids, probs_t = policy.exact_distribution(aryl, target_index)
                for row_id, probability in zip(row_ids, probs_t.cpu().numpy()):
                    joint[algorithm, policy.catalog.reactions[row_id].ligand] += (
                        0.5 * probability / condition_count
                    )
    p_algorithm = joint.sum(axis=1, keepdims=True)
    p_ligand = joint.sum(axis=0, keepdims=True)
    product = p_algorithm @ p_ligand
    mask = joint > 0
    mutual_information = float(np.sum(joint[mask] * np.log(joint[mask] / product[mask])))
    h_algorithm = float(-np.sum(p_algorithm[p_algorithm > 0] * np.log(p_algorithm[p_algorithm > 0])))
    h_ligand = float(-np.sum(p_ligand[p_ligand > 0] * np.log(p_ligand[p_ligand > 0])))
    if h_algorithm == 0 or h_ligand == 0:
        return 0.0
    return mutual_information / math.sqrt(h_algorithm * h_ligand)
