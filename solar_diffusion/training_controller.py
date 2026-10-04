from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class ControllerAction:
    rule: str
    old_val: float
    new_val: float
    reason: str = ""


def ctrl_actions_to_record(actions: Iterable[ControllerAction]) -> Dict[str, Any]:
    actions = list(actions)
    return {
        "controller_actions": [
            {"rule": a.rule, "old": a.old_val, "new": a.new_val, "reason": a.reason}
            for a in actions
        ],
        "controller_action_count": len(actions),
    }


class TrainingController:
    def __init__(self, cfg: dict, stage: str):
        if stage not in {"vae", "diff"}:
            raise ValueError("stage must be 'vae' or 'diff'")
        self.cfg = cfg
        self.stage = stage
        self._last_epoch = -1

    def _set_if_changed(self, section: dict, key: str, value: float, rule: str, reason: str):
        old = float(section.get(key, value))
        if value == old:
            return None
        section[key] = value
        return ControllerAction(rule, old, float(value), reason)

    def step_vae(self, epoch: int, train_metrics: dict, val_metrics: dict, diagnostics: dict) -> list[ControllerAction]:
        self._last_epoch = epoch
        if not self.cfg.get("controller", {}).get("enabled", True):
            return []
        section = self.cfg["training"]["vae"]
        actions = []
        grad = float(train_metrics.get("grad_norm", 0.0))
        threshold = float(self.cfg.get("diagnostics", {}).get("grad_norm_warn", 10.0))
        if grad > threshold:
            current = float(section.get("grad_clip", 1.0))
            floor = float(self.cfg.get("controller", {}).get("grad_clip_floor_vae", 0.5))
            factor = float(self.cfg.get("controller", {}).get("grad_clip_decay", 0.7))
            action = self._set_if_changed(section, "grad_clip", max(floor, current * factor), "grad_clip", "gradient norm exceeded the configured warning threshold")
            if action:
                actions.append(action)
        return actions

    def step_diff(self, epoch: int, train_metrics: dict, val_metrics: dict, gen_diag: dict | None = None, tau_diag: dict | None = None) -> list[ControllerAction]:
        self._last_epoch = epoch
        if not self.cfg.get("controller", {}).get("enabled", True):
            return []
        section = self.cfg["training"]["diffusion"]
        actions = []
        grad = float(train_metrics.get("grad_norm", 0.0))
        threshold = float(self.cfg.get("diagnostics", {}).get("grad_norm_warn", 10.0))
        if grad > threshold:
            current = float(section.get("grad_clip", 1.0))
            floor = float(self.cfg.get("controller", {}).get("grad_clip_floor_diff", 0.1))
            factor = float(self.cfg.get("controller", {}).get("grad_clip_decay", 0.7))
            action = self._set_if_changed(section, "grad_clip", max(floor, current * factor), "grad_clip", "gradient norm exceeded the configured warning threshold")
            if action:
                actions.append(action)
        return actions
