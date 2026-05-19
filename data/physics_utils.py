
# data/physics_utils.py
# ---------------------
# Assembles physics conditioning tensors from profile dicts produced by
# SolarPreprocessor.

# ETR strategy 
# ------------------------------
# ETR is a pure function of DOY — we compute it once per profile using a
# fast vectorised Spencer formula (no pvlib DatetimeIndex call in the hot
# path). The formula requires only integer DOY, so it runs in microseconds.

# Day features
# -----------------------------
# - Zenith threshold always comes from the caller, never hard-coded.
# - sunrise_hour / sunset_hour expressed as hours-from-solar-noon (signed),
#   not raw UTC hours.  This removes the latitude×timezone confound.


import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pvlib  # used only for declination_spencer71 scalar

log = logging.getLogger(__name__)

# Column names 
_ZENITH_COL = "solar_zenith_angle"
_GCS_COL    = "ghi_clear_sky"

# Day-feature indices
_DAY_FEAT_NAMES = [
    "declination",
    "doy_sin",
    "doy_cos",
    "sunrise_from_noon",   # hours before solar noon (negative = before)
    "sunset_from_noon",    # hours after solar noon (positive = after)
    "day_length_hours",
]


# Fast ETR
def _etr_spencer(doy: int, n_steps: int, dt_hours: float, solar_constant: float = 1367.0) -> np.ndarray:
    
    # Extraterrestrial radiation for every timestep of a day using Spencer (1971).

    # Returns (n_steps,) float32 in W/m2.
    # Each value is the mean ETR over the timestep interval (midpoint rule),
    # identical to pvlib's get_extra_radiation(method='spencer') behaviour.

    # Parameters
    # ----------
    # doy     : day-of-year (1–365)
    # n_steps : number of timesteps in the day (e.g. 144 for 10-min)
    # dt_hours: timestep width in hours (e.g. 1/6 for 10-min)
    
    B = (2 * math.pi / 365) * (doy - 1)
    E0 = 1.000110 + 0.034221 * math.cos(B) + 0.001280 * math.sin(B) \
       + 0.000719 * math.cos(2 * B) + 0.000077 * math.sin(2 * B)
    return np.full(n_steps, solar_constant * E0, dtype=np.float32)


def _dt_hours_from_profile(profile: Dict) -> float:
    """Infer timestep width in hours from the profile timestamps list."""
    ts = pd.to_datetime(profile["timestamps"][:2])
    return (ts[1] - ts[0]).total_seconds() / 3600.0


#Intra-day physics matrix

def extract_intraday_matrix(profile: Dict) -> np.ndarray:
    
    # Assemble the (T, 3) intra-day physics matrix.

    # Columns : [solar_zenith_angle, ETR, ghi_clear_sky]

    # ETR is computed via the fast Spencer formula - no pvlib DatetimeIndex
    
    geo = profile["deterministic_geometry"]
    try:
        zenith = np.array(geo[_ZENITH_COL], dtype=np.float32)
        gcs    = np.array(geo[_GCS_COL],    dtype=np.float32)
    except KeyError as e:
        raise KeyError(
            f"Missing geometry column {e} in profile {profile.get('date')}. "
            f"Available keys: {list(geo.keys())}"
        )

    T        = len(zenith)
    doy      = int(pd.Timestamp(profile["date"]).day_of_year)
    dt_hours = _dt_hours_from_profile(profile)
    etr      = _etr_spencer(doy, T, dt_hours)

    return np.stack([zenith, etr, gcs], axis=1)  # (T, 3)


# Sunlit mask 
def extract_sunlit_mask(
    profile: Dict,
    zenith_threshold: float,   # always explicit - no default to avoid inconsistency
) -> np.ndarray:
    """Boolean mask (T,) — True where solar zenith < threshold."""
    zenith = np.array(
        profile["deterministic_geometry"][_ZENITH_COL], dtype=np.float32
    )
    return zenith < zenith_threshold


# Day-level scalar features
def extract_day_features(
    profile: Dict,
    zenith_threshold: float,   # same value used for sunlit mask
) -> np.ndarray:
    
    # Compute the 7-dim day-level scalar conditioning vector.

    # Features
    # --------
    # 0  declination           solar declination [degrees] — Spencer eq.
    # 1  doy_sin               sin(2π·DOY/365)
    # 2  doy_cos               cos(2π·DOY/365)
    # 3  sunrise_from_noon     hours from solar noon to sunrise (negative)
    # 4  sunset_from_noon      hours from solar noon to sunset (positive)
    # 5  day_length_hours      total sunlit duration

    # TOA daily irradiation removed — it is a deterministic function of DOY
    # and latitude, both already encoded via doy_sin/cos and sunrise/sunset.
    # Its log z-score std was 0.024 causing z-scores of ±20, dominating gradients.

    # Sunrise/sunset are expressed relative to solar noon so the feature
    # is independent of UTC timezone offset
    # Zenith threshold matches the mask used everywhere else 
    
    geo      = profile["deterministic_geometry"]
    date_str = profile["date"]

    ts  = pd.Timestamp(date_str)
    doy = int(ts.day_of_year)
    doy_sin = float(np.sin(2 * math.pi * doy / 365.0))
    doy_cos = float(np.cos(2 * math.pi * doy / 365.0))
    declination = float(pvlib.solarposition.declination_spencer71(doy))

    # Derive sunrise/sunset relative to solar noon 
    # P3-FIX: all time arithmetic is done in UTC fractional hours throughout.
    # profile["solar_noon"] is an ISO string that may be tz-aware (UTC) or
    # tz-naive.  We strip tz information and treat everything as UTC wall-clock
    # hours, which is consistent because profile timestamps are also UTC-naive
    # after preprocessing resampling.
    zenith_arr = np.array(geo[_ZENITH_COL], dtype=np.float32)
    timestamps = pd.to_datetime(profile["timestamps"])            # tz-naive UTC

    assert len(timestamps) >= 2, (
        f"Profile {profile.get('date')} has fewer than 2 timestamps — cannot infer dt"
    )
    dt_hours = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0
    assert dt_hours > 0, f"Non-positive dt_hours={dt_hours} for profile {profile.get('date')}"

    # Solar noon: stored as ISO string in profile root
    noon_str = profile.get("solar_noon", None)

    # P3-FIX continued: sunlit_times is also tz-naive - subtraction is safe.
    # BUGFIX: do NOT use .hour/.minute on sunlit_times - for western-US stations
    # (UTC-6 to UTC-7) sunset falls after UTC midnight (00:00–01:30 of the next
    # calendar day), so .hour returns 0 or 1 instead of ~24, making
    # sunset_from_noon hugely negative (observed mean≈-5.78 instead of +5.5).
    # Fix: express all times as fractional hours elapsed since timestamps[0],
    # which is always within the local solar day regardless of UTC offset.
    t0 = timestamps[0]   # reference epoch - first sample of the day window

    def _frac_hours(ts: pd.Timestamp) -> float:
    # Fractional hours elapsed since t0 (monotonically increasing, no wrap)
        return (ts - t0).total_seconds() / 3600.0

    # Recompute noon_hour in the same elapsed-hours frame
    if noon_str:
        noon_ts = pd.Timestamp(noon_str)
        noon_ts = noon_ts.tz_localize(None) if noon_ts.tzinfo is not None else noon_ts
        noon_hour = _frac_hours(noon_ts)
    else:
        # Fallback already computed above; convert to elapsed-hours frame
        # by re-deriving from the sunlit midpoint timestamp
        sunlit_idx_fb = np.where(zenith_arr < zenith_threshold)[0]
        if len(sunlit_idx_fb) > 0:
            noon_hour = _frac_hours(timestamps[sunlit_idx_fb[len(sunlit_idx_fb) // 2]])
        else:
            noon_hour = _frac_hours(t0) + 12.0   # polar-night safe default

    sunlit_times = timestamps[zenith_arr < zenith_threshold]
    if len(sunlit_times) > 0:
        sr_hour = _frac_hours(sunlit_times[0])
        ss_hour = _frac_hours(sunlit_times[-1])
    else:
        # P12-FIX: empty sunlit window (polar night or extreme overcast).
        # Return safe zero-duration day centred on solar noon.
        sr_hour = noon_hour
        ss_hour = noon_hour

    sunrise_from_noon = sr_hour - noon_hour   # negative (before noon)
    sunset_from_noon  = ss_hour - noon_hour   # positive (after noon)

    if "day_length" in geo:
        day_length_hours = float(np.array(geo["day_length"], dtype=np.float32)[0])
    else:
        day_length_hours = max(0.0, ss_hour - sr_hour)

    result = np.array([
        declination, doy_sin, doy_cos,
        sunrise_from_noon, sunset_from_noon,
        day_length_hours,
    ], dtype=np.float32)

    assert np.all(np.isfinite(result)), (
        f"Non-finite day_features for profile {profile.get('date')}: {result}"
    )
    return result


# Location features 
def extract_location_features(lat: float, lon: float) -> np.ndarray:
    """Encode lat/lon as (4,) sin/cos vector."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    return np.array([
        np.sin(lat_r), np.cos(lat_r),
        np.sin(lon_r), np.cos(lon_r),
    ], dtype=np.float32)


# Climate features from kgcpy
# Enriches the denoiser's day-level conditioning with climatological context
# that pvlib solar geometry alone cannot provide.  Computed entirely from
# lat/lon via kgcpy - no historical observations needed - so available at
# inference time for any arbitrary location.
#
# kgcpy (pip install kgcpy) provides:
#   kgcpy.lookupCZ(lat, lon)          -> Köppen zone string e.g. "Cfa", "BWh"
#   kgcpy.irradianceQuantile(zone)    -> (p98, p80, p50, p30) tuple [Wh/m²/year]
#
# Feature vector (10 dims):
#   0-5   Köppen one-hot: tropical(A), arid(B), temperate(C),
#                         continental(D), polar(E), other/unknown
#   6-9   Irradiance quantiles (p30, p50, p80, p98) in Wh/m²/year
#         — normalised by ClimateFeatNormStats at training time
#
# Appended to 7-dim solar day features → 17 dims total per day.
# Config: diffusion.day_feat_dim must equal 7 + 10 = 17.
#
# INFERENCE CONTRACT: extract_climate_features(lat, lon) is called once per
# location and broadcast to all days.  No training data, no observations needed.
# Falls back to zeros gracefully if kgcpy is unavailable.

_KOPPEN_MAIN_GROUPS = {
    "A": 0,   # tropical
    "B": 1,   # arid
    "C": 2,   # temperate
    "D": 3,   # continental
    "E": 4,   # polar
}
_N_KOPPEN_CLASSES = 6    # 5 main + 1 unknown
_N_IRR_QUANTILES  = 4    # p30, p50, p80, p98
N_CLIMATE_FEATURES = _N_KOPPEN_CLASSES + _N_IRR_QUANTILES   # 10


def _koppen_to_onehot(zone_str: str) -> np.ndarray:
    # Map Köppen zone string (e.g. 'Cfb') to (_N_KOPPEN_CLASSES,) one-hot
    vec = np.zeros(_N_KOPPEN_CLASSES, dtype=np.float32)
    if zone_str and len(zone_str) >= 1:
        idx = _KOPPEN_MAIN_GROUPS.get(zone_str[0].upper(), _N_KOPPEN_CLASSES - 1)
    else:
        idx = _N_KOPPEN_CLASSES - 1   # unknown
    vec[idx] = 1.0
    return vec


def extract_climate_features(lat: float, lon: float) -> np.ndarray:
    # Compute (N_CLIMATE_FEATURES=10,) climate vector from lat/lon alone.

    # Features
    # --------
    # 0-5   Köppen one-hot (tropical, arid, temperate, continental, polar, other)
    # 6-9   Irradiance quantiles (p30, p50, p80, p98) in Wh/m2/year
    #       from kgcpy.irradianceQuantile -per-zone annual GHI statistics
    #       fitted on real historical irradiance data globally.

    # Falls back to zeros with a warning if kgcpy is not installed.
    # The model trains correctly without climate features; the denoiser will
    # learn climate conditioning only when kgcpy is available.
    
    koppen_vec = np.zeros(_N_KOPPEN_CLASSES, dtype=np.float32)
    irr_q      = np.zeros(_N_IRR_QUANTILES,  dtype=np.float32)
    zone_str   = "unknown"

    try:
        import kgcpy as _kgcpy

        # Köppen zone lookup — returns string like "Cfa", "BWh", "Dfc"
        zone_str   = str(_kgcpy.lookupCZ(lat, lon))
        koppen_vec = _koppen_to_onehot(zone_str)

        # Irradiance quantiles for this zone.
        # irradianceQuantile returns (p98, p80, p50, p30) — reorder to ascending.
        q_tuple = _kgcpy.irradianceQuantile(zone_str)
        if isinstance(q_tuple, tuple) and len(q_tuple) == 4:
            p98, p80, p50, p30 = (float(v) for v in q_tuple)
            irr_q = np.array([p30, p50, p80, p98], dtype=np.float32)
        else:
            log.warning(
                "kgcpy.irradianceQuantile returned unexpected value for zone %s: %s",
                zone_str, q_tuple,
            )

    except ImportError:
        log.warning(
            "kgcpy not installed — climate features will be zeros. "
            "Install with: pip install kgcpy  "
            "Training will proceed without climate conditioning."
        )
    except Exception as e:
        log.warning(
            "kgcpy lookup failed for (lat=%.4f, lon=%.4f): %s — "
            "climate features will be zeros.", lat, lon, e,
        )

    features = np.concatenate([koppen_vec, irr_q])   # (10,)
    log.debug(
        "Climate features (lat=%.2f lon=%.2f): zone=%s  "
        "p30=%.0f p50=%.0f p80=%.0f p98=%.0f Wh/m²/yr",
        lat, lon, zone_str,
        float(irr_q[0]), float(irr_q[1]), float(irr_q[2]), float(irr_q[3]),
    )
    return features


# Climate feature names for logging
_CLIMATE_FEAT_NAMES = [
    "koppen_tropical", "koppen_arid", "koppen_temperate",
    "koppen_continental", "koppen_polar", "koppen_other",
    "irr_p30", "irr_p50", "irr_p80", "irr_p98",
]
assert len(_CLIMATE_FEAT_NAMES) == N_CLIMATE_FEATURES

@dataclass
class _NormStats:
    mean_: np.ndarray = field(default_factory=lambda: np.array([]))
    std_:  np.ndarray = field(default_factory=lambda: np.array([]))
    _fitted: bool = field(default=False, repr=False)

    def fit(self, arrays: List[np.ndarray]) -> None:
        data = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        self.mean_ = data.mean(axis=0).astype(np.float32)
        self.std_  = data.std(axis=0).astype(np.float32)
        self.std_  = np.where(self.std_ < 1e-6, 1.0, self.std_).astype(np.float32)
        self._fitted = True
        log.info("NormStats fitted | mean=%s | std=%s",
                 self.mean_.round(3), self.std_.round(3))

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        self._check()
        return (arr.astype(np.float32) - self.mean_) / self.std_

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        self._check()
        return arr.astype(np.float32) * self.std_ + self.mean_

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, f, indent=2)
        log.info("NormStats saved → %s", path)

    def load(self, path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NormStats file not found: {path.resolve()}")
        with path.open() as f:
            blob = json.load(f)
        self.mean_   = np.array(blob["mean"], dtype=np.float32)
        self.std_    = np.array(blob["std"],  dtype=np.float32)
        self._fitted = True
        log.info("NormStats loaded ← %s", path)

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() or .load() before normalising.")


class IntraDayNormStats(_NormStats):
    pass

class DayFeatureNormStats(_NormStats):
    pass


class ClimateFeatNormStats(_NormStats):
    #Normalisation stats for the N_CLIMATE_FEATURES-dim climate feature vector.

    # Köppen one-hot dims (0 to _N_KOPPEN_CLASSES-1) are binary - passed through
    # unchanged as {0,1}.  Only the irradiance quantile dims are z-scored.
    # Mirrors the ObsPhysNormStats binary/continuous split pattern.
    # 

    def __init__(self):
        super().__init__()
        # Set binary_mask as instance attribute (not class-level) to avoid
        # shared-mutable-default issues across instances.
        self.binary_mask: np.ndarray = np.array(
            [True] * _N_KOPPEN_CLASSES + [False] * _N_IRR_QUANTILES, dtype=bool
        )

    def fit(self, arrays: List[np.ndarray]) -> None:
        """Fit z-score stats on irradiance quantile dims only."""
        if not arrays:
            self.mean_   = np.zeros(_N_IRR_QUANTILES, dtype=np.float32)
            self.std_    = np.ones(_N_IRR_QUANTILES,  dtype=np.float32)
            self._fitted = True
            return
        # Each array is (1, N_CLIMATE_FEATURES) - slice irradiance dims
        continuous = [a[..., _N_KOPPEN_CLASSES:] for a in arrays]
        data = np.concatenate(continuous, axis=0).reshape(-1, _N_IRR_QUANTILES)
        self.mean_ = data.mean(axis=0).astype(np.float32)
        self.std_  = data.std(axis=0).astype(np.float32)
        self.std_  = np.where(self.std_ < 1e-6, 1.0, self.std_).astype(np.float32)
        self._fitted = True
        log.info(
            "ClimateFeatNormStats fitted | irr_q mean=%s | irr_q std=%s",
            self.mean_.round(1), self.std_.round(1),
        )

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        # Z-score irradiance quantile dims; pass through Köppen one-hot unchanged.

        # After z-scoring, irradiance quantile dims are soft-clamped to [-3, +3].
        # This prevents out-of-distribution locations (e.g. desert or polar sites
        # not seen during training) from producing extreme feature values that the
        # denoiser has never encountered. 3-sigma covers 99.7% of the training
        # distribution — values beyond are gently clamped rather than extrapolating.
        # Köppen one-hot dims (0-5) are already {0,1} and need no clamping.
        # 
        self._check()
        out = arr.astype(np.float32).copy()
        # Z-score irradiance quantile dims
        out[..., _N_KOPPEN_CLASSES:] = (
            (out[..., _N_KOPPEN_CLASSES:] - self.mean_) / self.std_
        )
        # Soft clamp irradiance dims to [-3, +3] — OOD protection
        out[..., _N_KOPPEN_CLASSES:] = np.clip(
            out[..., _N_KOPPEN_CLASSES:], -3.0, 3.0
        )
        return out

    def save(self, path) -> None:
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, f, indent=2)
        log.info("ClimateFeatNormStats saved → %s", path)

    def load(self, path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ClimateFeatNormStats not found: {path.resolve()}")
        with path.open() as f:
            blob = json.load(f)
        self.mean_   = np.array(blob["mean"], dtype=np.float32)
        self.std_    = np.array(blob["std"],  dtype=np.float32)
        self._fitted = True
        log.info("ClimateFeatNormStats loaded ← %s", path)

# Maps profile dict key -> human-readable name (used for logging only).
# Top-level profile keys (measurements) + derived_features sub-dict keys
# (computed by SolarFeatureEngineer in preprocessing).
_OBS_CHANNEL_PROFILE_KEYS: Dict[str, str] = {
    # ── top-level measurement channels ───────────────────────────────────────
    "diffuse_fraction": "diffuse fraction (DIF/GHI)",
    "air_mass":         "air mass (optical path length)",
    "dni":              "DNI (beam normal irradiance)",
    # ── derived_features: variability ────────────────────────────────────────
    "csi_variability_w3":       "CSI rolling std (3-step window)",
    "csi_variability_w5":       "CSI rolling std (5-step window)",
    "csi_variability_w10":      "CSI rolling std (10-step window)",
    "ghi_variability_w3":       "GHI rolling std (3-step window)",
    "ghi_variability_w5":       "GHI rolling std (5-step window)",
    "ghi_variability_w10":      "GHI rolling std (10-step window)",
    "df_variability_w3":        "DF rolling std (3-step window)",
    "df_variability_w5":        "DF rolling std (5-step window)",
    # ── derived_features: trends ──────────────────────────────────────────────
    "csi_trend_w3":             "CSI linear trend (3-step)",
    "csi_trend_w5":             "CSI linear trend (5-step)",
    "csi_trend_w10":            "CSI linear trend (10-step)",
    "ghi_trend_w3":             "GHI linear trend (3-step)",
    "ghi_trend_w5":             "GHI linear trend (5-step)",
    "ghi_trend_w10":            "GHI linear trend (10-step)",
    "df_trend_w3":              "DF linear trend (3-step)",
    "df_trend_w5":              "DF linear trend (5-step)",
    # ── derived_features: ramp rates & events ────────────────────────────────
    "csi_ramp_rate":            "CSI instantaneous ramp rate",
    "ghi_ramp_rate":            "GHI instantaneous ramp rate",
    "dif_ramp_rate":            "DIF instantaneous ramp rate",
    "df_ramp_rate":             "DF instantaneous ramp rate",
    "irradiance_gradient_mag":  "combined irradiance gradient magnitude",
    "ghi_ramp_event_50":        "GHI ramp event flag (50 W/m²)",
    "ghi_ramp_event_100":       "GHI ramp event flag (100 W/m²)",
    "ghi_ramp_event_200":       "GHI ramp event flag (200 W/m²)",
    "ghi_ramp_event_300":       "GHI ramp event flag (300 W/m²)",
    "ghi_ramp_event_400":       "GHI ramp event flag (400 W/m²)",
    "ghi_ramp_event_500":       "GHI ramp event flag (500 W/m²)",
    "dif_ramp_event_50":        "DIF ramp event flag (50 W/m²)",
    "dif_ramp_event_100":       "DIF ramp event flag (100 W/m²)",
    "dif_ramp_event_200":       "DIF ramp event flag (200 W/m²)",
    "combined_ramp_event_50":   "combined ramp event flag (50 W/m²)",
    "combined_ramp_event_100":  "combined ramp event flag (100 W/m²)",
    "combined_ramp_event_200":  "combined ramp event flag (200 W/m²)",
    "combined_ramp_event_300":  "combined ramp event flag (300 W/m²)",
    "combined_ramp_event_400":  "combined ramp event flag (400 W/m²)",
    "combined_ramp_event_500":  "combined ramp event flag (500 W/m²)",
    # ── derived_features: persistence & cloud state ───────────────────────────
    "persistence_lag_1":        "CSI persistence autocorr (lag 1)",
    "persistence_lag_2":        "CSI persistence autocorr (lag 2)",
    "persistence_lag_3":        "CSI persistence autocorr (lag 3)",
    "csi_autocorrelation":      "CSI lag-1 autocorrelation (full day)",
    "sky_clarity_index":        "sky clarity index (clear-sky fraction proxy)",
    "cloud_enhancement":        "cloud-enhancement event flag",
}


# Per-channel pre-transforms applied BEFORE z-score normalisation 
# Problem: ObsPhysNormStats uses z-score (subtract mean, divide by std).
# This works well for symmetric, unimodal distributions but BREAKS for:
#
#   (a) Heavy right-skewed continuous channels (irradiance_gradient_mag):
#       mean=42, std=54, max=818 → after z-score, outlier at +14σ.
#       The Conv1d encoder learns to ignore the channel or gets unstable
#       gradients from the rare extreme values.
#       Fix: log1p(x) compresses the tail → near-normal distribution.
#
#   (b) Binary {0,1} event flags (combined_ramp_event_*):
#       At threshold 100: mean=0.07, std=0.25 → 0→-0.28, 1→+3.72
#       At threshold 300: mean=0.01, std=0.09 → 0→-0.11, 1→+11.0
#       At threshold 500: mean=0.00, std=0.03 → 0→ 0.00, 1→+33.3
#       z-score on binary flags produces extreme asymmetric outliers that
#       destabilize encoder training.  Binary flags must NOT be z-scored —
#       they are passed through as {0, 1} float32 unchanged.
#
# The pretransform is applied inside extract_obs_matrix so that
# ObsPhysNormStats.fit() and .normalize() see the already-transformed
# distribution.  Binary channels are tagged so fit_all_norm_stats and
# dataset.py can exclude them from the z-score fitting/application step.
#
# "identity" → no transform, z-score applied normally downstream
# "log1p"    → np.log1p(x) applied here; z-score applied downstream
# "binary"   → {0,1} float, NO z-score applied at any stage
_OBS_PRETRANSFORM: Dict[str, str] = {
    # ── top-level measurement channels ───────────────────────────────────
    "diffuse_fraction":         "identity",   # [0,1] symmetric enough — z-score fine
    "air_mass":                 "identity",   # [1,38] mildly skewed — z-score acceptable
    "dni":                      "log1p",
    # ── continuous derived: heavy right skew → log1p ──────────────────────
    "irradiance_gradient_mag":  "log1p",      # mean=42 std=54 max=818 → outlier at 14σ
    # ── binary event flags: NEVER z-score ────────────────────────────────
    # Ramp event flags at multiple thresholds — each a {0,1} indicator.
    # z-score would put the rare positive class at 3–33σ, destabilising
    # encoder training.  Passed through as {0,1} float32 unchanged.
    "ghi_ramp_event_50":        "binary",
    "ghi_ramp_event_100":       "binary",
    "ghi_ramp_event_200":       "binary",
    "ghi_ramp_event_300":       "binary",
    "ghi_ramp_event_400":       "binary",
    "ghi_ramp_event_500":       "binary",
    "dif_ramp_event_50":        "binary",
    "dif_ramp_event_100":       "binary",
    "dif_ramp_event_200":       "binary",
    "combined_ramp_event_50":   "binary",
    "combined_ramp_event_100":  "binary",
    "combined_ramp_event_200":  "binary",
    "combined_ramp_event_300":  "binary",
    "combined_ramp_event_400":  "binary",
    "combined_ramp_event_500":  "binary",
    "cloud_enhancement":        "binary",
    # ── continuous derived: well-behaved → z-score fine ──────────────────
    "csi_variability_w3":       "log1p",
    "csi_variability_w5":       "log1p",
    "csi_variability_w10":      "log1p",
    "ghi_variability_w3":       "log1p",
    "ghi_variability_w5":       "log1p",
    "ghi_variability_w10":      "log1p",
    "df_variability_w3":        "log1p",
    "df_variability_w5":        "log1p",
    "csi_trend_w3":             "identity",
    "csi_trend_w5":             "identity",
    "csi_trend_w10":            "identity",
    "ghi_trend_w3":             "identity",
    "ghi_trend_w5":             "identity",
    "ghi_trend_w10":            "identity",
    "df_trend_w3":              "identity",
    "df_trend_w5":              "identity",
    "csi_ramp_rate":            "identity",
    "ghi_ramp_rate":            "identity",
    "dif_ramp_rate":            "identity",
    "df_ramp_rate":             "identity",
    "persistence_lag_1":        "identity",
    "persistence_lag_2":        "identity",
    "persistence_lag_3":        "identity",
    "csi_autocorrelation":      "identity",
    "sky_clarity_index":        "identity",
}


def is_obs_channel_binary(name: str) -> bool:
    #Return True if channel 'name' is a binary {0,1} flag that must NOT be z-scored
    return _OBS_PRETRANSFORM.get(name, "identity") == "binary"


def obs_binary_mask(obs_channel_names: List[str]) -> np.ndarray:
    # 
    # Boolean array of shape (N_obs,) — True where the channel is binary.

    # Used in dataset.py and fit_all_norm_stats to exclude binary columns
    # from z-score fitting and application.

    # Example
    # -------
    # mask = obs_binary_mask(cfg["vae"]["obs_channels"])
    # # fit only on non-binary columns:
    # stats.fit(obs_arrays[:, ~mask])
    # # apply z-score only to non-binary columns:
    # obs_norm[:, ~mask] = stats.normalize(obs_raw[:, ~mask])
    # # binary columns stay as {0,1}:
    # obs_norm[:, mask] = obs_raw[:, mask]
    # 
    return np.array([is_obs_channel_binary(n) for n in obs_channel_names], dtype=bool)


def _apply_obs_pretransform(arr: np.ndarray, name: str) -> np.ndarray:
    # 
    # Apply per-channel pre-transform to a 1-D sunlit slice BEFORE z-score.

    # Called inside extract_obs_matrix for each column individually.
    # After this transform, the column is ready for ObsPhysNormStats.fit()
    # and .normalize() — except binary channels which bypass z-score entirely.

    # Parameters
    # ----------
    # arr  : (T_sun,) raw float32 values for one channel
    # name : channel name — looked up in _OBS_PRETRANSFORM

    # Returns
    # -------
    # (T_sun,) float32 transformed values
    # 
    mode = _OBS_PRETRANSFORM.get(name, "identity")
    arr  = arr.astype(np.float32)
    if mode == "log1p":
        # Compress heavy right tail: log1p(x) = log(1+x), defined at x=0.
        # Guard: clip negative values to 0 first (sensor noise may produce tiny negatives).
        return np.log1p(np.clip(arr, 0.0, None))
    # "identity" and "binary": return unchanged.
    # Binary {0,1} values pass through here and are later excluded from z-score.
    return arr


def extract_obs_matrix(
    profile: Dict,
    obs_channel_names: List[str],
    zenith_threshold: float,
) -> np.ndarray:
    # 
    # Assemble the (T_sun, N_obs) observation matrix from a profile dict.

    # These channels are derived from REAL MEASUREMENTS or from
    # SolarFeatureEngineer (stored in profile["derived_features"]).
    # They are only available during training (when actual irradiance data
    # exists).  At generation time this function is never called — the
    # encoder is never invoked; diffusion directly samples z in τ-space.

    # Lookup order for each channel name
    # -----------------------------------
    # 1. profile[name]                      — top-level keys (diffuse_fraction,
    #                                         air_mass, dni, ...)
    # 2. profile["derived_features"][name]  — SolarFeatureEngineer output
    #                                         (csi_variability_w3, ghi_trend_w5, ...)
    # 3. Missing → fill 0.0 + log warning

    # Pre-transform
    # -------------
    # Each channel is passed through _apply_obs_pretransform() before being
    # stored.  This applies log1p to heavy-skewed channels and is a no-op for
    # well-behaved and binary channels.  See _OBS_PRETRANSFORM for the full
    # per-channel policy.

    # Binary channels ({0,1} event flags) are stored as-is here.
    # The caller (dataset.py / fit_all_norm_stats) must use obs_binary_mask()
    # to exclude them from z-score fitting and application.

    # Parameters
    # ----------
    # profile            : profile dict produced by SolarPreprocessor
    # obs_channel_names  : ordered list of channel names — from cfg["vae"]["obs_channels"]
    # zenith_threshold   : same value used everywhere else for the sunlit mask

    # Returns
    # -------
    # obs : (T_sun, N_obs) float32 — only sunlit timesteps, pre-transformed,
    #       same order as obs_channel_names.
    #       Binary columns are {0,1}; continuous columns are pretransformed.
    #       Z-score normalisation is applied DOWNSTREAM by ObsPhysNormStats
    #       on NON-BINARY columns only.
    # 
    zenith = np.array(
        profile["deterministic_geometry"][_ZENITH_COL], dtype=np.float32
    )
    sunlit = zenith < zenith_threshold   # (T,) bool

    T_sun   = int(sunlit.sum())
    mat     = np.zeros((T_sun, len(obs_channel_names)), dtype=np.float32)
    derived = profile.get("derived_features", {})   # sub-dict from SolarFeatureEngineer

    for col_idx, name in enumerate(obs_channel_names):
        # Priority 1: top-level profile key (measurements: diffuse_fraction, air_mass, dni)
        raw = profile.get(name, None)
        # Priority 2: derived_features sub-dict (SolarFeatureEngineer outputs)
        if raw is None:
            raw = derived.get(name, None)
        if raw is None:
            log.warning(
                "Observation channel '%s' missing from profile %s "
                "(checked top-level keys and derived_features) — filling with 0.0",
                name, profile.get("date", "?"),
            )
            continue
        full_arr        = np.array(raw, dtype=np.float32)          # (T,)
        sunlit_arr      = full_arr[sunlit]                          # (T_sun,)
        mat[:, col_idx] = _apply_obs_pretransform(sunlit_arr, name) # pre-transform

    return mat   # (T_sun, N_obs) — pre-transformed, binary cols still {0,1}


class ObsPhysNormStats(_NormStats):
    # 
    # Normalisation statistics for the observation-channel matrix.

    # Fitted on training profiles only, identical API to IntraDayNormStats.
    # Saved/loaded from cfg["paths"]["norm_stats_obs_phys"].

    # Extends _NormStats.save/load to also persist binary_mask and
    # obs_channel_names, which are set dynamically after fit() in
    # fit_all_norm_stats().  Without persisting these, load() returns an
    # object missing binary_mask, so dataset.py falls into the else-branch
    # and passes the full (T_sun, N_obs) array to normalize() whose
    # mean_/std_ have shape (N_continuous,) — a broadcast error at runtime.
    # 

    def __init__(self):
        super().__init__()
        self.binary_mask:       Optional[np.ndarray] = None
        self.obs_channel_names: Optional[List[str]]  = None

    def save(self, path) -> None:
        """Save mean_, std_, binary_mask, and obs_channel_names to JSON."""
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean_.tolist(),
            "std":  self.std_.tolist(),
        }
        if self.binary_mask is not None:
            payload["binary_mask"] = self.binary_mask.tolist()
        if self.obs_channel_names is not None:
            payload["obs_channel_names"] = list(self.obs_channel_names)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("NormStats saved → %s", path)

    def load(self, path) -> None:
        """Load mean_, std_, binary_mask, and obs_channel_names from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NormStats file not found: {path.resolve()}")
        with path.open() as f:
            blob = json.load(f)
        self.mean_   = np.array(blob["mean"], dtype=np.float32)
        self.std_    = np.array(blob["std"],  dtype=np.float32)
        self._fitted = True
        if "binary_mask" in blob:
            self.binary_mask = np.array(blob["binary_mask"], dtype=bool)
        if "obs_channel_names" in blob:
            self.obs_channel_names = list(blob["obs_channel_names"])
        log.info("NormStats loaded ← %s", path)


# 6. Fit both stats from a list of profiles 
def fit_all_norm_stats_from_cfg(
    profiles: List[Dict],
    cfg: Dict,
) -> tuple:
    # Convenience wrapper: reads all parameters from cfg and calls fit_all_norm_stats.

    # This is the recommended call site in main.py — all thresholds and channel
    # names are read from config automatically.

    # Returns (intraday_stats, day_feat_stats, obs_stats, climate_stats).
    # 
    return fit_all_norm_stats(
        profiles=profiles,
        zenith_threshold=cfg["data"]["zenith_threshold_deg"],
        subsample=1,
        obs_channel_names=cfg["vae"].get("obs_channels", []),
    )


def fit_all_norm_stats(
    profiles: List[Dict],
    zenith_threshold: float,
    subsample: int = 1,
    obs_channel_names: Optional[List[str]] = None,
) -> Tuple["IntraDayNormStats", "DayFeatureNormStats", Optional["ObsPhysNormStats"], "ClimateFeatNormStats"]:
    # 
    # Fit all normalisation objects from a list of profile dicts.

    # Returns
    # -------
    # intraday_stats : IntraDayNormStats   — for (zenith, ETR, GCS) sequences
    # day_feat_stats : DayFeatureNormStats — for 7-dim solar day-level features
    # obs_stats      : ObsPhysNormStats | None
    # climate_stats  : ClimateFeatNormStats — for N_CLIMATE_FEATURES=10-dim climate features
    #                  (Köppen one-hot + irradiance quantiles from kgcpy)

    # Fitted on training profiles only — never val/test.
    # 
    obs_channel_names = obs_channel_names or []

    log.info(
        "Fitting normalisation stats from %d profiles (subsample=%d) "
        "| obs_channels=%s …",
        len(profiles), subsample, obs_channel_names or "none",
    )
    intraday_arrays:  List[np.ndarray] = []
    day_feat_arrays:  List[np.ndarray] = []
    obs_arrays:       List[np.ndarray] = []
    climate_arrays:   List[np.ndarray] = []

    # Cache climate features per unique (lat, lon) to avoid redundant lookups.
    # Each kgcpy lookup can take ~0.5s — caching saves minutes for large datasets.
    _climate_cache: Dict[tuple, np.ndarray] = {}

    for profile in profiles[::subsample]:
        matrix = extract_intraday_matrix(profile)
        zenith_col = np.array(
            profile["deterministic_geometry"][_ZENITH_COL], dtype=np.float32
        )
        sunlit_mask = zenith_col < zenith_threshold
        matrix_sunlit = matrix[sunlit_mask]
        if len(matrix_sunlit) > 0:
            intraday_arrays.append(matrix_sunlit)

        day_feat_arrays.append(extract_day_features(profile, zenith_threshold))

        if obs_channel_names:
            obs_mat = extract_obs_matrix(profile, obs_channel_names, zenith_threshold)
            obs_arrays.append(obs_mat)

        # Climate features: compute once per unique (lat, lon) pair
        lat = float(profile.get("lat", 0.0))
        lon = float(profile.get("lon", 0.0))
        key = (round(lat, 4), round(lon, 4))
        if key not in _climate_cache:
            _climate_cache[key] = extract_climate_features(lat, lon)
        climate_arrays.append(_climate_cache[key].reshape(1, -1))

    intraday_stats = IntraDayNormStats()
    intraday_stats.fit(intraday_arrays)

    day_feat_stats = DayFeatureNormStats()
    day_feat_stats.fit(day_feat_arrays)

    # Climate stats — fit on irradiance quantile dims only (Köppen is binary)
    climate_stats = ClimateFeatNormStats()
    climate_stats.fit(climate_arrays)

    obs_stats: Optional[ObsPhysNormStats] = None
    if obs_channel_names and obs_arrays:
        b_mask       = obs_binary_mask(obs_channel_names)
        n_binary     = int(b_mask.sum())
        n_continuous = int((~b_mask).sum())

        obs_stats = ObsPhysNormStats()

        if n_continuous > 0:
            continuous_arrays = [a[:, ~b_mask] for a in obs_arrays]
            obs_stats.fit(continuous_arrays)
        else:
            obs_stats.mean_   = np.array([], dtype=np.float32)
            obs_stats.std_    = np.array([], dtype=np.float32)
            obs_stats._fitted = True

        obs_stats.binary_mask       = b_mask
        obs_stats.obs_channel_names = list(obs_channel_names)

        log.info(
            "ObsPhysNormStats fitted | N_obs=%d (%d continuous z-scored, "
            "%d binary pass-through) | channels=%s",
            len(obs_channel_names), n_continuous, n_binary, obs_channel_names,
        )
        if n_binary > 0:
            binary_names = [n for n, b in zip(obs_channel_names, b_mask) if b]
            log.info("  Binary channels (NOT z-scored): %s", binary_names)

    log.info(
        "Climate features fitted | %d unique locations | %d Köppen classes + %d irr quantiles",
        len(_climate_cache), _N_KOPPEN_CLASSES, _N_IRR_QUANTILES,
    )

    return intraday_stats, day_feat_stats, obs_stats, climate_stats