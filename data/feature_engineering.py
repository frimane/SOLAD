from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


class SolarFeatureEngineer:
    def __init__(self, config: dict):
        section = config.get("feature_engineering", {})
        data_section = config.get("data", {}).get("feature_engineering", {})
        self.config = section or data_section or {}
        self.enabled = bool(self.config.get("enabled", False))
        self._features = tuple(self.config.get("features", ())) if self.enabled else ()

    def get_feature_list(self) -> list[str]:
        return list(self._features)

    def compute_all_features(self, profile: Dict[str, Iterable[float]]) -> Dict[str, np.ndarray]:
        if not self.enabled:
            return {}
        ghi = np.asarray(profile["ghi"], dtype=np.float32)
        dif = np.asarray(profile["dif"], dtype=np.float32)
        csi = np.asarray(profile["csi"], dtype=np.float32)
        diffuse_fraction = np.asarray(profile["diffuse_fraction"], dtype=np.float32)
        gradients = np.gradient(csi).astype(np.float32)
        candidates = {
            "irradiance_gradient_mag": np.abs(np.gradient(ghi)).astype(np.float32),
            "csi_ramp_rate": np.abs(gradients),
            "csi_trend_w5": np.convolve(csi, np.ones(5, dtype=np.float32) / 5, mode="same"),
            "diffuse_fraction": diffuse_fraction,
            "ghi": ghi,
            "dif": dif,
            "csi": csi,
        }
        missing = [name for name in self._features if name not in candidates]
        if missing:
            raise KeyError(f"Unsupported derived features: {', '.join(missing)}")
        return {name: candidates[name] for name in self._features}
