# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learning-rate and stopping decisions based on complete held-out epochs."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real

_GOOD_DESCENT_RATE = 0.8
_RELATIVE_FLOOR = 1e-12


def _selection_metric(validation) -> float | None:
    """Return the eligible, finite checkpoint-selection metric or None."""
    selection = validation.get("selection") if isinstance(validation, Mapping) else None
    if not isinstance(selection, Mapping) or not selection.get("eligible"):
        return None
    metric = selection.get("metric")
    if isinstance(metric, bool) or not isinstance(metric, Real) or not math.isfinite(metric):
        return None
    return float(metric)


class PlateauController:
    """Distinguish useful convergence from stalled training and the epoch cap.

    Decisions use the validator's checkpoint-selection metric
    ``validation["selection"]["metric"]`` (mean final free-corner force residual
    norm [N], lower is better) and only when ``validation["selection"]["eligible"]``
    is true; ineligible or nonfinite validations count as bad epochs and never
    improve the best value. An improvement must be relative:
    ``best - metric >= threshold * max(best, 1e-12)``. A "good" epoch is a
    complete one with every physical trajectory surviving and a descent rate of
    at least 0.8. Stopping requires ``allow_early_stop``, the minimum epoch
    count, ``stop_patience`` bad epochs and at least two learning-rate
    reductions; it reports ``plateau_converged`` when the last five epochs were
    good and ``stalled`` otherwise. Only validated epochs are observed: epochs
    that skipped validation are absent from the sequence and count neither as
    improvements nor against patience. ``reset_metric_history`` forgets the
    best metric and patience when the validation budget changes. ``best_loss``
    keeps its attribute name for checkpoint compatibility but stores the best
    selection metric.
    """

    def __init__(
        self,
        learning_rate=1e-4,
        min_epochs=30,
        max_epochs=200,
        lr_patience=5,
        stop_patience=15,
        threshold=1e-3,
        min_lr=1e-6,
    ):
        if not (0 < min_lr <= learning_rate and 1 <= min_epochs <= max_epochs):
            raise ValueError("invalid learning rates or epoch limits")
        if lr_patience < 1 or stop_patience < 1 or threshold <= 0:
            raise ValueError("patience and improvement threshold must be positive")
        self.learning_rate = learning_rate
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.lr_patience = lr_patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_loss = None
        self.bad_epochs = 0
        self.lr_bad_epochs = 0
        self.reductions = 0
        self.last_epoch = 0
        self.recent_good = []

    def observe(self, epoch: int, validation: dict, *, allow_early_stop: bool = True) -> dict:
        """Consume validation, retaining learning-rate reductions when stopping is disabled.

        Args:
            epoch: One-based training epoch of this validation, greater than
                the previously observed epoch.
            validation: Summary from ``mixed_validation.validate`` with
                ``selection``, ``physical_survivors``, ``sample_count`` and
                ``descent_rate``.
            allow_early_stop: Whether plateau or stall stopping may trigger;
                the epoch limit applies regardless.

        Returns:
            ``{"learning_rate", "stop", "status"}`` with status ``running``,
            ``plateau_converged``, ``stalled`` or ``epoch_limit``.
        """
        if epoch <= self.last_epoch:
            raise ValueError("validation epochs must increase")
        self.last_epoch = epoch
        metric = _selection_metric(validation)
        complete = metric is not None
        improved = complete and (
            self.best_loss is None or self.best_loss - metric >= self.threshold * max(self.best_loss, _RELATIVE_FLOOR)
        )
        if improved:
            self.best_loss = metric
            self.bad_epochs = self.lr_bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.lr_bad_epochs += 1
        if self.lr_bad_epochs >= self.lr_patience:
            new_lr = max(self.min_lr, self.learning_rate * 0.5)
            self.reductions += int(new_lr < self.learning_rate)
            self.learning_rate = new_lr
            self.lr_bad_epochs = 0
        descent = validation.get("descent_rate")
        good = (
            complete
            and validation.get("physical_survivors") == validation.get("sample_count")
            and isinstance(descent, Real)
            and not isinstance(descent, bool)
            and descent >= _GOOD_DESCENT_RATE
        )
        self.recent_good = [*self.recent_good, bool(good)][-5:]
        status = "running"
        if (
            allow_early_stop
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.stop_patience
            and self.reductions >= 2
        ):
            status = "plateau_converged" if len(self.recent_good) == 5 and all(self.recent_good) else "stalled"
        elif epoch >= self.max_epochs:
            status = "epoch_limit"
        return {"learning_rate": self.learning_rate, "stop": status != "running", "status": status}

    def reset_metric_history(self) -> None:
        """Forget the best metric, patience counters and recent-quality history.

        The learning rate, its reduction count and the last observed epoch are
        kept. Call this when the validation budget changes on resume so that
        metrics measured under the new budget are not compared with the old
        best.
        """
        self.best_loss = None
        self.bad_epochs = self.lr_bad_epochs = 0
        self.recent_good = []

    def state_dict(self) -> dict:
        """Return a small, serializable controller state."""
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state: dict):
        """Restore history while retaining the caller's permitted new epoch cap."""
        max_epochs = self.max_epochs
        if set(state) != set(vars(self)):
            raise ValueError("incompatible plateau controller state")
        self.__dict__.update(copy.deepcopy(state))
        self.max_epochs = max_epochs
