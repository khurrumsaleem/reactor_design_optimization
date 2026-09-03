from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import ReactionPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def measured_reward(measured_yield: float, target: float, objective: str) -> float:
    if objective == "target_match":
        return -abs(measured_yield - target) / 100.0
    if objective == "maximize":
        return measured_yield / 100.0
    raise ValueError(f"unknown objective: {objective}")


def atomic_torch_save(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _global_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_global_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _mle_epoch(
    policy: ReactionPolicy,
    examples: Sequence[tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> float:
    shuffled = list(examples)
    random.shuffle(shuffled)
    losses: list[float] = []
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start : start + batch_size]
        optimizer.zero_grad()
        loss = -torch.stack(
            [
                policy.log_prob(policy.catalog.reactions[row_id].aryl, target_token, row_id)
                for row_id, target_token in batch
            ]
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _train_mle(
    phase: str,
    policy: ReactionPolicy,
    examples: Sequence[tuple[int, int]],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    checkpoint_path: str | Path | None,
    resume: bool,
) -> list[float]:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    start_epoch = 0
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if resume and checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=policy.device, weights_only=False)
        if state.get("phase") != phase or state.get("total_epochs") != epochs:
            raise ValueError(f"checkpoint/config mismatch: {checkpoint}")
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["next_epoch"])
        losses = list(state["losses"])
        _restore_global_rng(state["global_rng"])
        print(f"[{phase}] resuming at epoch {start_epoch}/{epochs}", flush=True)

    def save(next_epoch: int, complete: bool) -> None:
        if checkpoint:
            atomic_torch_save(
                {
                    "phase": phase,
                    "total_epochs": epochs,
                    "next_epoch": next_epoch,
                    "complete": complete,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                    "global_rng": _global_rng_state(),
                },
                checkpoint,
            )

    if checkpoint and not checkpoint.exists():
        save(0, False)
    next_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            losses.append(_mle_epoch(policy, examples, optimizer, batch_size))
            next_epoch = epoch + 1
            save(next_epoch, next_epoch == epochs)
            print(f"[{phase}] epoch {next_epoch}/{epochs}, loss={losses[-1]:.4f}", flush=True)
    except KeyboardInterrupt:
        save(next_epoch, False)
        print(f"[{phase}] interrupted; checkpoint saved at epoch {next_epoch}", flush=True)
        raise
    return losses


def train_cpt(
    policy: ReactionPolicy,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> list[float]:
    examples = [(row_id, 0) for row_id in policy.catalog.restricted_row_ids]
    return _train_mle(
        "cpt", policy, examples, epochs, learning_rate, batch_size, checkpoint_path, resume
    )


def train_sft(
    policy: ReactionPolicy,
    targets: Sequence[float],
    examples_per_task_target: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> list[float]:
    examples: list[tuple[int, int]] = []
    for aryl in range(policy.catalog.n_aryls):
        for target_index, target in enumerate(targets, start=1):
            examples.extend(
                (row_id, target_index)
                for row_id in policy.catalog.closest_rows(
                    aryl, target, examples_per_task_target, restricted=True
                )
            )
    return _train_mle(
        "sft", policy, examples, epochs, learning_rate, batch_size, checkpoint_path, resume
    )


def _balanced_target_schedule(steps: int, n_targets: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        block = list(range(n_targets))
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:steps]


def _online_checkpoint_payload(
    algorithm: str,
    policy: ReactionPolicy,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    generator: torch.Generator,
    history: list[dict[str, object]],
    next_step: int,
    total_steps: int,
    seed: int,
    complete: bool,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "total_steps": total_steps,
        "seed": seed,
        "next_step": next_step,
        "complete": complete,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": rng.getstate(),
        "torch_generator": generator.get_state(),
        "history": history,
    }


def _restore_online(
    checkpoint: Path,
    algorithm: str,
    policy: ReactionPolicy,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    generator: torch.Generator,
    steps: int,
    seed: int,
) -> tuple[int, list[dict[str, object]]]:
    state = torch.load(checkpoint, map_location=policy.device, weights_only=False)
    if (
        state.get("algorithm") != algorithm
        or state.get("total_steps") != steps
        or state.get("seed") != seed
    ):
        raise ValueError(f"checkpoint/config mismatch: {checkpoint}")
    policy.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.setstate(state["python_rng"])
    generator.set_state(state["torch_generator"])
    return int(state["next_step"]), list(state["history"])


def online_dpo(
    policy: ReactionPolicy,
    targets: Sequence[float],
    steps: int,
    learning_rate: float,
    beta: float,
    temperature: float,
    objective: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = False,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    history: list[dict[str, object]] = []
    start_step = 0
    if resume and checkpoint and checkpoint.exists():
        start_step, history = _restore_online(
            checkpoint, "dpo", policy, optimizer, rng, generator, steps, seed
        )
        print(f"[dpo] resuming at step {start_step}/{steps}", flush=True)
    schedule = _balanced_target_schedule(steps, len(targets), seed + 17)

    def save(next_step: int, complete: bool) -> None:
        if checkpoint:
            atomic_torch_save(
                _online_checkpoint_payload(
                    "dpo", policy, optimizer, rng, generator, history,
                    next_step, steps, seed, complete,
                ),
                checkpoint,
            )

    if checkpoint and not checkpoint.exists():
        save(0, False)
    next_step = start_step
    try:
        for step in range(start_step, steps):
            aryl = rng.randrange(policy.catalog.n_aryls)
            target_index = schedule[step]
            target_token = target_index + 1
            target = float(targets[target_index])
            first = policy.sample(aryl, target_token, temperature, generator)
            second = policy.sample(aryl, target_token, temperature, generator)
            rf = measured_reward(policy.catalog.reactions[first].measured_yield, target, objective)
            rs = measured_reward(policy.catalog.reactions[second].measured_yield, target, objective)
            if rf == rs:
                loss_value = 0.0
            else:
                chosen, rejected = (first, second) if rf > rs else (second, first)
                optimizer.zero_grad()
                margin = policy.log_prob(aryl, target_token, chosen) - policy.log_prob(
                    aryl, target_token, rejected
                )
                loss = -F.logsigmoid(beta * margin)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach())
            for sample_index, (row_id, reward) in enumerate(((first, rf), (second, rs))):
                row = policy.catalog.reactions[row_id]
                history.append(
                    {
                        "step": step,
                        "sample": sample_index,
                        "aryl_id": aryl,
                        "target": target,
                        "row_id": row_id,
                        "measured_yield": row.measured_yield,
                        "reward": reward,
                        "expanded": not policy.catalog.is_in_training_support(row),
                        "loss": loss_value,
                    }
                )
            next_step = step + 1
            if next_step % checkpoint_every == 0 or next_step == steps:
                save(next_step, next_step == steps)
                print(f"[dpo] step {next_step}/{steps}", flush=True)
    except KeyboardInterrupt:
        save(next_step, False)
        print(f"[dpo] interrupted; checkpoint saved at step {next_step}", flush=True)
        raise
    return history


def online_grpo(
    policy: ReactionPolicy,
    targets: Sequence[float],
    steps: int,
    group_size: int,
    learning_rate: float,
    temperature: float,
    objective: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = False,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    history: list[dict[str, object]] = []
    start_step = 0
    if resume and checkpoint and checkpoint.exists():
        start_step, history = _restore_online(
            checkpoint, "grpo", policy, optimizer, rng, generator, steps, seed
        )
        print(f"[grpo] resuming at step {start_step}/{steps}", flush=True)
    schedule = _balanced_target_schedule(steps, len(targets), seed + 17)

    def save(next_step: int, complete: bool) -> None:
        if checkpoint:
            atomic_torch_save(
                _online_checkpoint_payload(
                    "grpo", policy, optimizer, rng, generator, history,
                    next_step, steps, seed, complete,
                ),
                checkpoint,
            )

    if checkpoint and not checkpoint.exists():
        save(0, False)
    next_step = start_step
    try:
        for step in range(start_step, steps):
            aryl = rng.randrange(policy.catalog.n_aryls)
            target_index = schedule[step]
            target_token = target_index + 1
            target = float(targets[target_index])
            samples = [
                policy.sample(aryl, target_token, temperature, generator)
                for _ in range(group_size)
            ]
            rewards = torch.tensor(
                [
                    measured_reward(policy.catalog.reactions[i].measured_yield, target, objective)
                    for i in samples
                ],
                dtype=torch.float32,
                device=policy.device,
            )
            std = rewards.std(unbiased=False)
            if float(std) < 1e-8:
                loss_value = 0.0
            else:
                advantages = (rewards - rewards.mean()) / (std + 1e-8)
                optimizer.zero_grad()
                log_probs = torch.stack(
                    [policy.log_prob(aryl, target_token, row_id) for row_id in samples]
                )
                loss = -(advantages.detach() * log_probs).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach())
            for sample_index, (row_id, reward) in enumerate(zip(samples, rewards.tolist())):
                row = policy.catalog.reactions[row_id]
                history.append(
                    {
                        "step": step,
                        "sample": sample_index,
                        "aryl_id": aryl,
                        "target": target,
                        "row_id": row_id,
                        "measured_yield": row.measured_yield,
                        "reward": reward,
                        "expanded": not policy.catalog.is_in_training_support(row),
                        "loss": loss_value,
                    }
                )
            next_step = step + 1
            if next_step % checkpoint_every == 0 or next_step == steps:
                save(next_step, next_step == steps)
                print(f"[grpo] step {next_step}/{steps}", flush=True)
    except KeyboardInterrupt:
        save(next_step, False)
        print(f"[grpo] interrupted; checkpoint saved at step {next_step}", flush=True)
        raise
    return history


def clone_policy(policy: ReactionPolicy) -> ReactionPolicy:
    return copy.deepcopy(policy)
