# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learning-rate and stopping decisions based on complete held-out epochs."""

from __future__ import annotations

import copy
import math


class PlateauController:
    """Distinguish useful convergence from stalled training and the epoch cap."""

    def __init__(
        self,
        learning_rate=1e-4,
        min_epochs=30,
        max_epochs=200,
        lr_patience=5,
        stop_patience=15,
        threshold=1e-4,
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
        """Consume validation, retaining learning-rate reductions when stopping is disabled."""
        if epoch != self.last_epoch + 1:
            raise ValueError("validation epochs must be consecutive and start at one")
        self.last_epoch = epoch
        loss = validation.get("mean_normalized_loss")
        complete = validation.get("failed_count") == 0 and loss is not None and math.isfinite(loss)
        improved = complete and (self.best_loss is None or self.best_loss - loss >= self.threshold)
        if improved:
            self.best_loss = loss
            self.bad_epochs = self.lr_bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.lr_bad_epochs += 1
        if self.lr_bad_epochs >= self.lr_patience:
            new_lr = max(self.min_lr, self.learning_rate * 0.5)
            self.reductions += int(new_lr < self.learning_rate)
            self.learning_rate = new_lr
            self.lr_bad_epochs = 0
        before, after = validation.get("mean_before_joule"), validation.get("mean_after_joule")
        good = (
            complete
            and validation.get("descent_rate", 0) >= 0.95
            and before is not None
            and after is not None
            and math.isfinite(before)
            and math.isfinite(after)
            and after < before
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
