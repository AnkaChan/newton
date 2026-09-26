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
    """Increase K/H caps after qualified validation or a configured epoch limit.

    Experimental. Stages expose caps (1,8), (2,16), (4,32), (8,64),
    (16,128), (32,128), retaining configured shorter counts. A stage advances
    when both its minimum epoch residence and consecutive qualifying-validation
    patience are satisfied and the full-horizon check passed, or the optional
    maximum residence is reached. Qualifying epochs during the minimum
    residence count toward patience. Advancing resets both counters;
    final-stage observations continue counting without advancing.

    Existing settings recorded explicitly: a 10-epoch minimum residence for
    validation-based advancement (``min_stage_epochs``), two consecutive
    qualifying validations (``patience``), the 0.8 inclusive descent-rate
    threshold (``min_descent_rate``) and the trainer's hard 20-epoch stage cap
    (``max_stage_epochs``; ``None`` here keeps uncapped legacy runs).

    Cheap validation must report no failures, finite decreasing mean physical
    energy, sufficient descent rate, and survival of every physical trajectory.
    Missing or nonfinite diagnostics do not qualify. Validation-gated
    advancement additionally requires the full available-horizon check
    (``observe(..., full_horizon=...)``) with no failures and every trajectory
    surviving; a missing or failed full check blocks that route and resets the
    qualifying streak, but never vetoes the hard cap. ``needs_full_horizon``
    tells the trainer when the next observation could advance so the check can
    run first. The component supplies available counts only: callers apply
    changes to new resets, leaving the K/H choices of active trajectories
    intact. Defaults are configurable implementation choices rather than fixed
    training-campaign settings.

    Args:
        iteration_counts: Available positive inner-iteration counts, including 1.
        physical_step_counts: Available positive trajectory lengths with at
            least one value no greater than 8.
        min_stage_epochs: Minimum validation epochs spent in each stage.
        max_stage_epochs: Optional hard residence limit, at least the minimum.
            Reaching this limit advances even if validation does not qualify.
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
        max_stage_epochs=None,
        patience=2,
        min_descent_rate=0.8,
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
        if max_stage_epochs is not None and not _integer(max_stage_epochs, minimum=min_stage_epochs):
            raise ValueError("max_stage_epochs must be None or an integer at least min_stage_epochs")
        if not _finite(min_descent_rate) or not 0 <= min_descent_rate <= 1:
            raise ValueError("min_descent_rate must be finite and in [0,1]")
        self.min_stage_epochs = int(min_stage_epochs)
        self.max_stage_epochs = int(max_stage_epochs) if max_stage_epochs is not None else None
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

    @property
    def final_stage(self) -> int:
        """Return the index of the last stage (K=32/H=128), after which caps never change."""
        return len(self._CAPS) - 1

    def _qualifies(self, validation) -> bool:
        """Return whether one cheap validation summary qualifies for advancement."""
        if not isinstance(validation, dict):
            return False
        failed = validation.get("failed_count")
        count = validation.get("sample_count")
        survivors = validation.get("physical_survivors")
        before = validation.get("mean_before_joule")
        after = validation.get("mean_after_joule")
        descent = validation.get("descent_rate")
        return bool(
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

    @staticmethod
    def _full_horizon_qualifies(full_horizon) -> bool:
        """Return whether a full-horizon summary reports no failures and full survival."""
        if not isinstance(full_horizon, dict):
            return False
        failed = full_horizon.get("failed_count")
        count = full_horizon.get("sample_count")
        survivors = full_horizon.get("physical_survivors")
        return bool(
            _integer(failed)
            and failed == 0
            and _integer(count, minimum=1)
            and _integer(survivors)
            and survivors == count
        )

    def needs_full_horizon(self, validation: dict) -> bool:
        """Return whether observing ``validation`` next could advance the stage.

        True when the observation would satisfy the validation gate (qualified,
        residence and patience) or reach the hard residence limit, and the
        current stage is not final. Callers run the full-horizon check before
        ``observe`` in that case.
        """
        if self.stage >= self.final_stage:
            return False
        by_validation = (
            self._qualifies(validation)
            and self.stage_epochs + 1 >= self.min_stage_epochs
            and self.qualified_epochs + 1 >= self.patience
        )
        by_cap = self.max_stage_epochs is not None and self.stage_epochs + 1 >= self.max_stage_epochs
        return bool(by_validation or by_cap)

    def observe(self, validation: dict, *, full_horizon: dict | None = None) -> dict:
        """Consume one complete epoch's validation and report current progress.

        Expected keys are ``failed_count``, ``sample_count``,
        ``physical_survivors``, ``mean_before_joule``, ``mean_after_joule`` and
        ``descent_rate``. ``full_horizon`` is the summary of
        ``validate_full_horizon`` when it ran this epoch; validation-gated
        advancement requires it with ``failed_count == 0`` and
        ``physical_survivors == sample_count``. When the gate is otherwise
        satisfied but the full check is missing or failed, the stage does not
        advance by validation and ``qualified_epochs`` resets to zero. The hard
        residence limit advances regardless. Returned counters describe the
        current stage after any advancement. ``qualified`` describes this
        observation, ``full_horizon_qualified`` the full check (None when not
        run), ``advanced`` indicates a stage transition and ``advance_reason``
        records whether validation or the hard residence limit caused it.
        """
        qualified = self._qualifies(validation)
        full_qualified = None if full_horizon is None else self._full_horizon_qualifies(full_horizon)
        self.stage_epochs += 1
        self.qualified_epochs = self.qualified_epochs + 1 if qualified else 0
        gate = (
            self.stage < self.final_stage
            and self.stage_epochs >= self.min_stage_epochs
            and self.qualified_epochs >= self.patience
        )
        advanced, reason = False, None
        if gate and full_qualified:
            self.stage += 1
            self.stage_epochs = self.qualified_epochs = 0
            advanced, reason = True, "validation"
        elif gate:
            self.qualified_epochs = 0
        if not advanced and self.advance_if_overdue():
            advanced, reason = True, "max_stage_epochs"
        return {
            "stage": self.stage,
            "advanced": bool(advanced),
            "advance_reason": reason,
            "qualified": bool(qualified),
            "full_horizon_qualified": full_qualified,
            "stage_epochs": self.stage_epochs,
            "qualified_epochs": self.qualified_epochs,
        }

    def advance_if_overdue(self) -> bool:
        """Apply a hard limit once without observing or reclassifying validation.

        This also applies a newly configured limit to restored progress. Extra
        residence in the previous stage never counts toward the next stage.
        """
        if (
            self.stage < self.final_stage
            and self.max_stage_epochs is not None
            and self.stage_epochs >= self.max_stage_epochs
        ):
            self.stage += 1
            self.stage_epochs = self.qualified_epochs = 0
            return True
        return False

    def state_dict(self) -> dict:
        """Return serializable configuration and progress without mutable aliases."""
        return {
            "version": 1,
            "iteration_counts": list(self.iteration_counts),
            "physical_step_counts": list(self.physical_step_counts),
            "min_stage_epochs": self.min_stage_epochs,
            "max_stage_epochs": self.max_stage_epochs,
            "patience": self.patience,
            "min_descent_rate": self.min_descent_rate,
            "stage": self.stage,
            "stage_epochs": self.stage_epochs,
            "qualified_epochs": self.qualified_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore validated configuration and counters atomically for exact resume."""
        if isinstance(state, dict):
            state = {"max_stage_epochs": None, **state}
        if not isinstance(state, dict) or set(state) != set(self.state_dict()) or state["version"] != 1:
            raise ValueError("incompatible mixed curriculum state")
        restored = MixedCurriculum(
            state["iteration_counts"],
            state["physical_step_counts"],
            min_stage_epochs=state["min_stage_epochs"],
            max_stage_epochs=state["max_stage_epochs"],
            patience=state["patience"],
            min_descent_rate=state["min_descent_rate"],
        )
        if (
            not _integer(state["stage"])
            or state["stage"] > self.final_stage
            or not _integer(state["stage_epochs"])
            or not _integer(state["qualified_epochs"])
            or state["qualified_epochs"] > state["stage_epochs"]
        ):
            raise ValueError("invalid mixed curriculum progress counters")
        restored.stage = int(state["stage"])
        restored.stage_epochs = int(state["stage_epochs"])
        restored.qualified_epochs = int(state["qualified_epochs"])
        self.__dict__.update(vars(restored))
