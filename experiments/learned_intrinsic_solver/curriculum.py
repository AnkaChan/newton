# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental validation-gated count caps for mixed-trajectory training."""

from __future__ import annotations

import math
from numbers import Integral, Real

__all__ = ["MixedCurriculum"]


def _integer(value, *, minimum=0) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class MixedCurriculum:
    """Increase K/H caps only after sufficient complete, stable validation.

    Experimental. Stages expose caps (1,8), (2,16), (4,32), (8,64),
    (16,128), (32,128), retaining configured shorter counts. A stage advances
    when both its minimum epoch residence and consecutive qualifying-validation
    patience are satisfied. Qualifying epochs during the minimum residence count
    toward patience. Advancing resets both counters; final-stage observations
    continue counting without advancing.

    Validation must report no failures, finite decreasing mean physical energy,
    sufficient descent rate, and survival of every physical trajectory. Missing
    or nonfinite diagnostics do not qualify. The component supplies available
    counts only: callers apply changes to new resets, leaving the K/H choices
    of active trajectories intact. Defaults are configurable implementation
    choices rather than fixed training-campaign settings.

    Args:
        iteration_counts: Available positive inner-iteration counts, including 1.
        physical_step_counts: Available positive trajectory lengths with at
            least one value no greater than 8.
        min_stage_epochs: Minimum validation epochs spent in each stage.
        patience: Required consecutive qualifying validation epochs.
        min_descent_rate: Inclusive qualifying descent-rate threshold in [0,1].
    """

    _CAPS = ((1, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 128))

    def __init__(
        self,
        iteration_counts=(1, 2, 4, 8, 16, 32),
        physical_step_counts=(8, 16, 32, 64, 128),
        *,
        min_stage_epochs=10,
        patience=2,
        min_descent_rate=0.9,
    ):
        def counts(values, name, maximum):
            values = tuple(values)
            if not values or any(not _integer(value, minimum=1) or value > maximum for value in values):
                raise ValueError(f"{name} must contain positive integers no greater than {maximum}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicate counts")
            return tuple(sorted(int(value) for value in values))

        self.iteration_counts = counts(iteration_counts, "iteration_counts", 32)
        self.physical_step_counts = counts(physical_step_counts, "physical_step_counts", 128)
        if 1 not in self.iteration_counts or self.physical_step_counts[0] > 8:
            raise ValueError("initial curriculum requires K=1 and an H <= 8")
        if not _integer(min_stage_epochs, minimum=1) or not _integer(patience, minimum=1):
            raise ValueError("min_stage_epochs and patience must be positive integers")
        if not _finite(min_descent_rate) or not 0 <= min_descent_rate <= 1:
            raise ValueError("min_descent_rate must be finite and in [0,1]")
        self.min_stage_epochs = int(min_stage_epochs)
        self.patience = int(patience)
        self.min_descent_rate = float(min_descent_rate)
        self.stage = 0
        self.stage_epochs = 0
        self.qualified_epochs = 0

    @property
    def available_counts(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return configured K/H choices under the current caps for new resets."""
        k_cap, h_cap = self._CAPS[self.stage]
        return (
            tuple(value for value in self.iteration_counts if value <= k_cap),
            tuple(value for value in self.physical_step_counts if value <= h_cap),
        )

    def observe(self, validation: dict) -> dict:
        """Consume one complete epoch's validation and report current progress.

        Expected keys are ``failed_count``, ``sample_count``,
        ``physical_survivors``, ``mean_before_joule``, ``mean_after_joule`` and
        ``descent_rate``. Returned counters describe the current stage after
        any advancement. ``qualified`` describes this observation, while
        ``advanced`` indicates a stage transition.
        """
        failed = validation.get("failed_count")
        count = validation.get("sample_count")
        survivors = validation.get("physical_survivors")
        before = validation.get("mean_before_joule")
        after = validation.get("mean_after_joule")
        descent = validation.get("descent_rate")
        qualified = (
            _integer(failed)
            and failed == 0
            and _integer(count, minimum=1)
            and _integer(survivors)
            and survivors == count
            and _finite(before)
            and _finite(after)
            and after < before
            and _finite(descent)
            and self.min_descent_rate <= descent <= 1
        )
        self.stage_epochs += 1
        self.qualified_epochs = self.qualified_epochs + 1 if qualified else 0
        advanced = (
            self.stage < len(self._CAPS) - 1
            and self.stage_epochs >= self.min_stage_epochs
            and self.qualified_epochs >= self.patience
        )
        if advanced:
            self.stage += 1
            self.stage_epochs = self.qualified_epochs = 0
        return {
            "stage": self.stage,
            "advanced": bool(advanced),
            "qualified": bool(qualified),
            "stage_epochs": self.stage_epochs,
            "qualified_epochs": self.qualified_epochs,
        }

    def state_dict(self) -> dict:
        """Return serializable configuration and progress without mutable aliases."""
        return {
            "version": 1,
            "iteration_counts": list(self.iteration_counts),
            "physical_step_counts": list(self.physical_step_counts),
            "min_stage_epochs": self.min_stage_epochs,
            "patience": self.patience,
            "min_descent_rate": self.min_descent_rate,
            "stage": self.stage,
            "stage_epochs": self.stage_epochs,
            "qualified_epochs": self.qualified_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore validated configuration and counters atomically for exact resume."""
        if not isinstance(state, dict) or set(state) != set(self.state_dict()) or state["version"] != 1:
            raise ValueError("incompatible mixed curriculum state")
        restored = MixedCurriculum(
            state["iteration_counts"],
            state["physical_step_counts"],
            min_stage_epochs=state["min_stage_epochs"],
            patience=state["patience"],
            min_descent_rate=state["min_descent_rate"],
        )
        if (
            not _integer(state["stage"])
            or state["stage"] >= len(self._CAPS)
            or not _integer(state["stage_epochs"])
            or not _integer(state["qualified_epochs"])
            or state["qualified_epochs"] > state["stage_epochs"]
        ):
            raise ValueError("invalid mixed curriculum progress counters")
        restored.stage = int(state["stage"])
        restored.stage_epochs = int(state["stage_epochs"])
        restored.qualified_epochs = int(state["qualified_epochs"])
        self.__dict__.update(vars(restored))
