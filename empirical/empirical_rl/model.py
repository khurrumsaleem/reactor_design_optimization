from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .data import ReactionCatalog


def _masked_logits(logits: torch.Tensor, allowed: Iterable[int]) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))
    ids = torch.tensor(tuple(allowed), dtype=torch.long, device=logits.device)
    mask[ids] = logits[ids]
    return mask


class ReactionPolicy(nn.Module):
    """Small autoregressive categorical policy with an empirical prefix trie."""

    def __init__(self, catalog: ReactionCatalog, n_targets: int, hidden_size: int = 48):
        super().__init__()
        self.catalog = catalog
        self.n_targets = n_targets
        h = hidden_size
        self.aryl_embedding = nn.Embedding(catalog.n_aryls, h)
        self.target_embedding = nn.Embedding(n_targets + 1, h)  # 0 is CPT/unconditional
        self.ligand_embedding = nn.Embedding(catalog.n_ligands, h)
        self.additive_embedding = nn.Embedding(catalog.n_additives, h)
        self.context = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh())
        self.ligand_head = nn.Linear(h, catalog.n_ligands)
        self.additive_head = nn.Linear(2 * h, catalog.n_additives)
        self.base_head = nn.Linear(3 * h, catalog.n_bases)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _context(self, aryl: int, target_token: int) -> torch.Tensor:
        a = torch.tensor(aryl, dtype=torch.long, device=self.device)
        t = torch.tensor(target_token, dtype=torch.long, device=self.device)
        return self.context(torch.cat([self.aryl_embedding(a), self.target_embedding(t)]))

    def component_logits(
        self, aryl: int, target_token: int, ligand: int | None = None, additive: int | None = None
    ) -> torch.Tensor:
        ctx = self._context(aryl, target_token)
        if ligand is None:
            return _masked_logits(self.ligand_head(ctx), self.catalog.valid_ligands[aryl])
        l = torch.tensor(ligand, dtype=torch.long, device=self.device)
        if additive is None:
            logits = self.additive_head(torch.cat([ctx, self.ligand_embedding(l)]))
            return _masked_logits(logits, self.catalog.valid_additives[(aryl, ligand)])
        ad = torch.tensor(additive, dtype=torch.long, device=self.device)
        logits = self.base_head(
            torch.cat([ctx, self.ligand_embedding(l), self.additive_embedding(ad)])
        )
        return _masked_logits(logits, self.catalog.valid_bases[(aryl, ligand, additive)])

    def log_prob(self, aryl: int, target_token: int, row_id: int) -> torch.Tensor:
        row = self.catalog.reactions[row_id]
        if row.aryl != aryl:
            raise ValueError("row belongs to a different aryl-halide task")
        lp_l = torch.log_softmax(self.component_logits(aryl, target_token), dim=-1)[row.ligand]
        lp_a = torch.log_softmax(
            self.component_logits(aryl, target_token, row.ligand), dim=-1
        )[row.additive]
        lp_b = torch.log_softmax(
            self.component_logits(aryl, target_token, row.ligand, row.additive), dim=-1
        )[row.base]
        return lp_l + lp_a + lp_b

    def sample(
        self,
        aryl: int,
        target_token: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> int:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        def draw(logits: torch.Tensor) -> int:
            probs = torch.softmax(logits / temperature, dim=-1)
            return int(torch.multinomial(probs, 1, generator=generator).item())

        with torch.no_grad():
            ligand = draw(self.component_logits(aryl, target_token))
            additive = draw(self.component_logits(aryl, target_token, ligand))
            base = draw(self.component_logits(aryl, target_token, ligand, additive))
        return self.catalog.lookup(aryl, ligand, additive, base).row_id

    def exact_distribution(self, aryl: int, target_token: int) -> tuple[list[int], torch.Tensor]:
        row_ids = self.catalog.rows_by_aryl[aryl]
        log_probs = torch.stack([self.log_prob(aryl, target_token, row_id) for row_id in row_ids])
        probs = torch.exp(log_probs)
        return row_ids, probs / probs.sum()

