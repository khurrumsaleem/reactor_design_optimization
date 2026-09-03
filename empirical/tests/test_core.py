from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from empirical_rl.data import ReactionCatalog
from empirical_rl.model import ReactionPolicy
from empirical_rl.training import measured_reward, online_dpo, online_grpo, set_seed


class EmpiricalCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = ReactionCatalog(ROOT / "data" / "buchwald_hartwig.csv")

    def test_cardinalities_and_row_count(self) -> None:
        self.assertEqual(len(self.catalog.reactions), 3955)
        self.assertEqual(
            (self.catalog.n_ligands, self.catalog.n_additives, self.catalog.n_bases, self.catalog.n_aryls),
            (4, 22, 3, 15),
        )

    def test_lookup_is_exact_and_missing_combination_fails(self) -> None:
        first = self.catalog.reactions[0]
        found = self.catalog.lookup(first.aryl, first.ligand, first.additive, first.base)
        self.assertEqual(found.row_id, first.row_id)
        all_designs = {
            (r.aryl, r.ligand, r.additive, r.base) for r in self.catalog.reactions
        }
        missing = next(
            design
            for design in (
                (a, l, d, b)
                for a in range(self.catalog.n_aryls)
                for l in range(self.catalog.n_ligands)
                for d in range(self.catalog.n_additives)
                for b in range(self.catalog.n_bases)
            )
            if design not in all_designs
        )
        with self.assertRaises(KeyError):
            self.catalog.lookup(*missing)

    def test_restricted_support_uses_all_ligands_but_not_all_pairs(self) -> None:
        restricted = [self.catalog.reactions[i] for i in self.catalog.restricted_row_ids]
        self.assertEqual({r.ligand for r in restricted}, set(range(4)))
        self.assertTrue(all(r.ligand == self.catalog.anchor_ligand(r.aryl) for r in restricted))
        self.assertTrue(any(not self.catalog.is_in_training_support(r) for r in self.catalog.reactions))

    def test_policy_samples_only_measured_rows(self) -> None:
        set_seed(7)
        policy = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
        generator = torch.Generator().manual_seed(7)
        for aryl in range(self.catalog.n_aryls):
            for _ in range(10):
                row_id = policy.sample(aryl, 1, generator=generator)
                self.assertEqual(self.catalog.reactions[row_id].aryl, aryl)

    def test_exact_distribution_normalizes(self) -> None:
        policy = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
        for aryl in range(self.catalog.n_aryls):
            _, probabilities = policy.exact_distribution(aryl, 1)
            self.assertAlmostEqual(float(probabilities.detach().sum()), 1.0, places=5)

    def test_short_online_updates_run(self) -> None:
        policy = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
        dpo_rows = online_dpo(policy, [80.0], 2, 1e-3, 0.1, 1.0, "target_match", 1)
        self.assertEqual(len(dpo_rows), 4)
        policy = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
        grpo_rows = online_grpo(policy, [80.0], 2, 3, 1e-3, 1.0, "target_match", 2)
        self.assertEqual(len(grpo_rows), 6)

    def test_completed_online_checkpoint_resumes_without_repeating_steps(self) -> None:
        set_seed(11)
        initial = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
        initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dpo.resume.pt"
            first = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
            first.load_state_dict(initial_state)
            original_history = online_dpo(
                first,
                [80.0],
                4,
                1e-3,
                0.01,
                1.0,
                "target_match",
                13,
                checkpoint_path=checkpoint,
                checkpoint_every=2,
            )
            resumed = ReactionPolicy(self.catalog, n_targets=1, hidden_size=12)
            resumed.load_state_dict(initial_state)
            resumed_history = online_dpo(
                resumed,
                [80.0],
                4,
                1e-3,
                0.01,
                1.0,
                "target_match",
                13,
                checkpoint_path=checkpoint,
                checkpoint_every=2,
                resume=True,
            )
            self.assertEqual(original_history, resumed_history)
            for name, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
