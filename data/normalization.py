#
# data/normalization.py
# ---------------------
# SolarBridge NormalizationManager.
# 
# This is a purpose-built replacement for the original 1152-line class.
# It keeps ONLY what SolarBridge actually calls:
# 
#   normalize_ghi_clearsky(arr)   -> G_norm(t) in [0, 1]   (network input ch-2)
#   denormalize_ghi_clearsky(arr) -> back to physical W/m^2
# 
#   normalize_csi(arr)            -> scaled CSI in [0, 1]   (NOT used for bridge K;
#   denormalize_csi(arr)            kept only for external callers / plotting)
# 
#   g_max     -> dataset-level maximum G_cs [W/m^2]  (used in bridge.py loss weight)
#   csi_min   -> dataset-level CSI minimum           (for external reference)
#   csi_max   -> dataset-level CSI maximum           (for external reference)
# 
# IMPORTANT: K (clear-sky index) fed to the bridge is RAW physical CSI.
# normalize_csi is NOT called in dataset.__getitem__ - it exists only for
# any downstream plotting or evaluation code that needs a [0,1] CSI.
# K_bar is also stored as raw CSI so bridge interpolation is consistent.
# 
# Stats are loaded from the JSON file saved by your existing
# NormalizationManager (same format: {"stats": {...}, "constant_features": [...]}).
# Nothing is recomputed here - the stats come from your upstream project.
# 
# What we read from the JSON
# --------------------------
#   stats['variables']['csi']          -> {method, min, max, range}
#   stats['geometry']['ghi_clear_sky'] -> {min, max, constant}
#   constant_features                   -> list of feature names
# 
# All other keys (dif, ghi, wavelet_deviations, derived, ...) are
# loaded but never touched.
#

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

log = logging.getLogger(__name__)

# The only two things SolarBridge normalises
_CSI_VAR  = "csi"
_GCS_FEAT = "ghi_clear_sky"


class SolarBridgeNorm:
    #
    # Thin normalisation wrapper for SolarBridge.
    # 
    # Usage:
    #     norm = SolarBridgeNorm(cfg)
    #     norm.load("data/profiles/normalization_stats.json")
    #     G_norm = norm.normalize_ghi_clearsky(g_raw_array)   # for network input
    #     g_max  = norm.g_max                                  # for bridge loss weight
    #

    def __init__(self, cfg: Dict[str, Any]):
        nc = cfg["normalization"]
        self.eps       = nc["epsilon"]
        self.min_range = nc["min_range_threshold"]

        # filled by load()
        self._csi_stats: Dict  = {}
        self._gcs_stats: Dict  = {}
        self._constant_features = set()
        self._loaded = False

    # -- I/O ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        #
        # Load stats saved by the original NormalizationManager.save().
        # Raises FileNotFoundError if the file is missing.
        # Raises KeyError if the required variable/feature stats are absent.
        #
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path.resolve()}")

        with path.open() as f:
            blob = json.load(f)

        stats = blob["stats"]
        self._constant_features = set(blob.get("constant_features", []))

        # -- CSI variable stats -----------------------------------------------
        if _CSI_VAR not in stats.get("variables", {}):
            raise KeyError(
                f"'{_CSI_VAR}' not found in stats['variables']. "
                f"Available: {list(stats.get('variables', {}).keys())}"
            )
        self._csi_stats = stats["variables"][_CSI_VAR]

        # -- ghi_clear_sky geometry stats -------------------------------------
        if _GCS_FEAT not in stats.get("geometry", {}):
            raise KeyError(
                f"'{_GCS_FEAT}' not found in stats['geometry']. "
                f"Available: {list(stats.get('geometry', {}).keys())}"
            )
        self._gcs_stats = stats["geometry"][_GCS_FEAT]

        self._loaded = True
        self._log_summary()

    def _log_summary(self) -> None:
        cs = self._csi_stats
        gs = self._gcs_stats
        log.info(
            "SolarBridgeNorm loaded | CSI: method=%s [%.4f, %.4f] | "
            "G_cs: [%.1f, %.1f] constant=%s | g_max=%.1f",
            cs.get("method", "?"),
            cs.get("min", float("nan")), cs.get("max", float("nan")),
            gs.get("min", float("nan")), gs.get("max", float("nan")),
            gs.get("constant", False),
            self.g_max,
        )

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call norm.load(path) before normalising.")

    # -- convenience properties -----------------------------------------------

    @property
    def g_max(self) -> float:
        #
        # Dataset-level maximum clear-sky irradiance [W/m^2].
        # Used in bridge.py to normalise the loss weight G_raw / g_max so
        # that the weight stays O(1) while preserving correct cross-sample
        # geographic weighting (unlike per-batch normalisation).
        #
        self._check_loaded()
        return float(self._gcs_stats["max"])

    @property
    def csi_min(self) -> float:
        #Dataset-level minimum raw CSI (for reference / plotting).
        self._check_loaded()
        return float(self._csi_stats["min"])

    @property
    def csi_max(self) -> float:
        #Dataset-level maximum raw CSI (for reference / plotting).
        self._check_loaded()
        return float(self._csi_stats["max"])

    # -- CSI (for plotting / external callers only - NOT used for bridge K) ---

    def normalize_csi(self, arr: np.ndarray) -> np.ndarray:
        #
        # CSI -> [0, 1] min-max.
        # 
        # NOTE: This is NOT called in dataset.__getitem__ for the bridge.
        # K fed to the bridge is raw physical CSI.  This method is kept
        # only for external callers (plotting, evaluation) that need a
        # normalised CSI representation.
        #
        self._check_loaded()
        s     = self._csi_stats
        vmin  = s["min"]
        vmax  = s["max"]
        denom = (vmax - vmin) + self.eps
        return np.clip((arr.astype(np.float32) - vmin) / denom, 0.0, 1.0)

    def denormalize_csi(self, arr: np.ndarray) -> np.ndarray:
        #[0, 1] -> physical CSI (inverse of normalize_csi).
        self._check_loaded()
        s = self._csi_stats
        return arr.astype(np.float32) * (s["max"] - s["min"]) + s["min"]

    # -- ghi_clear_sky --------------------------------------------------------

    def normalize_ghi_clearsky(self, arr: np.ndarray) -> np.ndarray:
        #
        # Raw ghi_clear_sky [W/m^2] -> [0, 1] using dataset-level min/max.
        # 
        # This is G_norm: the network input channel 1.  Normalisation uses
        # the dataset-level stats (not per-sample), so the relative magnitude
        # across different latitudes and seasons is preserved.
        # 
        # Returns 0.0 (not 0.5) when the feature is constant or degenerate:
        # a flat value of 0.5 would incorrectly gate night positions at 0.5
        # instead of 0, breaking the physics-derived soft gate in the U-Net.
        #
        self._check_loaded()
        s   = self._gcs_stats
        arr = arr.astype(np.float32)

        if s.get("constant", False) or _GCS_FEAT in self._constant_features:
            log.warning("ghi_clear_sky flagged as constant - returning zeros")
            return np.zeros_like(arr)

        vmin       = s["min"]
        feat_range = s["max"] - vmin

        if feat_range < self.min_range:
            log.warning("ghi_clear_sky range=%.2e < threshold - returning zeros",
                        feat_range)
            return np.zeros_like(arr)

        norm = (arr - vmin) / (feat_range + self.eps)
        return np.clip(norm, 0.0, 1.0)

    def denormalize_ghi_clearsky(self, arr: np.ndarray) -> np.ndarray:
        #[0, 1] -> physical ghi_clear_sky [W/m^2].
        self._check_loaded()
        s = self._gcs_stats
        return arr.astype(np.float32) * (s["max"] - s["min"]) + s["min"]