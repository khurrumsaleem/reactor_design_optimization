from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Reaction:
    row_id: int
    ligand: int
    additive: int
    base: int
    aryl: int
    measured_yield: float


class ReactionCatalog:
    """Exact finite catalog of measured reactions; never imputes missing rows."""

    fields = ("Ligand", "Additive", "Base", "Aryl halide")

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        if not raw:
            raise ValueError("reaction catalog is empty")
        expected = set(self.fields) | {"Output"}
        if set(raw[0]) != expected:
            raise ValueError(f"expected columns {sorted(expected)}, got {sorted(raw[0])}")

        self.values = {
            field: tuple(sorted({row[field] for row in raw})) for field in self.fields
        }
        self.to_id = {
            field: {value: index for index, value in enumerate(values)}
            for field, values in self.values.items()
        }
        self.reactions: list[Reaction] = []
        self._row_by_design: dict[tuple[int, int, int, int], int] = {}
        for row_id, row in enumerate(raw):
            reaction = Reaction(
                row_id=row_id,
                ligand=self.to_id["Ligand"][row["Ligand"]],
                additive=self.to_id["Additive"][row["Additive"]],
                base=self.to_id["Base"][row["Base"]],
                aryl=self.to_id["Aryl halide"][row["Aryl halide"]],
                measured_yield=float(row["Output"]),
            )
            design = (reaction.aryl, reaction.ligand, reaction.additive, reaction.base)
            if design in self._row_by_design:
                raise ValueError(f"duplicate measured design: {design}")
            self._row_by_design[design] = row_id
            self.reactions.append(reaction)

        self.rows_by_aryl: dict[int, list[int]] = {i: [] for i in range(self.n_aryls)}
        self.valid_ligands: dict[int, tuple[int, ...]] = {}
        self.valid_additives: dict[tuple[int, int], tuple[int, ...]] = {}
        self.valid_bases: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for reaction in self.reactions:
            self.rows_by_aryl[reaction.aryl].append(reaction.row_id)
        for aryl in range(self.n_aryls):
            rows = [self.reactions[i] for i in self.rows_by_aryl[aryl]]
            self.valid_ligands[aryl] = tuple(sorted({r.ligand for r in rows}))
            for ligand in self.valid_ligands[aryl]:
                lr = [r for r in rows if r.ligand == ligand]
                self.valid_additives[(aryl, ligand)] = tuple(sorted({r.additive for r in lr}))
                for additive in self.valid_additives[(aryl, ligand)]:
                    lar = [r for r in lr if r.additive == additive]
                    self.valid_bases[(aryl, ligand, additive)] = tuple(
                        sorted({r.base for r in lar})
                    )

    @property
    def n_ligands(self) -> int:
        return len(self.values["Ligand"])

    @property
    def n_additives(self) -> int:
        return len(self.values["Additive"])

    @property
    def n_bases(self) -> int:
        return len(self.values["Base"])

    @property
    def n_aryls(self) -> int:
        return len(self.values["Aryl halide"])

    def anchor_ligand(self, aryl: int) -> int:
        """Outcome-independent, deterministic, approximately balanced mapping."""
        return aryl % self.n_ligands

    def is_in_training_support(self, row_or_id: int | Reaction) -> bool:
        row = self.reactions[row_or_id] if isinstance(row_or_id, int) else row_or_id
        return row.ligand == self.anchor_ligand(row.aryl)

    @property
    def restricted_row_ids(self) -> list[int]:
        return [r.row_id for r in self.reactions if self.is_in_training_support(r)]

    def lookup(self, aryl: int, ligand: int, additive: int, base: int) -> Reaction:
        """Return a measured row or fail; there is intentionally no fallback."""
        key = (aryl, ligand, additive, base)
        try:
            return self.reactions[self._row_by_design[key]]
        except KeyError as exc:
            raise KeyError(f"unmeasured reaction combination: {key}") from exc

    def closest_rows(
        self,
        aryl: int,
        target_yield: float,
        k: int,
        restricted: bool = True,
    ) -> list[int]:
        candidates = self.rows_by_aryl[aryl]
        if restricted:
            candidates = [i for i in candidates if self.is_in_training_support(i)]
        return sorted(
            candidates,
            key=lambda i: (abs(self.reactions[i].measured_yield - target_yield), i),
        )[:k]

    def describe_row(self, row_id: int) -> dict[str, object]:
        row = self.reactions[row_id]
        return {
            "row_id": row_id,
            "aryl_id": row.aryl,
            "ligand_id": row.ligand,
            "additive_id": row.additive,
            "base_id": row.base,
            "measured_yield": row.measured_yield,
            "in_training_support": self.is_in_training_support(row),
        }

