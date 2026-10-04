#
# data/dataset.py
# ---------------
# SolarSequenceDataset - connects SolarPreprocessor output to the
# physics-conditioned latent diffusion model.
# 
# Design decisions
# ----------------
# - Variable-length sunlit slices:
#     Each day's K* and intra-day physics matrix is cropped to the sunlit
#     window (zenith < threshold). Lengths vary by season.
#     All sunlit slices in a batch are padded to the longest one in that batch;
#     a boolean valid_mask is returned so the VAE encoder ignores padding.
# 
# - Sequence windows:
#     A sample is W consecutive days from a single station's profile list.
#     Overlapping windows (stride=1) maximise data use.
#     Windows never cross station boundaries.
# 
# - Location conditioning:
#     lat/lon are encoded as a (4,) sin/cos vector.  The denoiser uses this
#     together with day_features to learn the climatological K* prior for
#     each (location, season) pair from training data - no regime labels needed.
# 
# - Sampling:
#     Climate-aware weighted sampling across all windows (training split only).
#     With multiple stations the true marginal distribution is dominated by
#     whichever stations have the highest clear-sky fraction (e.g. Desert Rock
#     is ~75% clear, Goodwin Creek ~45%).  Uniform sampling would therefore
#     under-represent overcast and mixed windows from cloudy stations and
#     over-represent clear windows from arid stations.
# 
#     _MultiStationDataset.get_sampling_weights() returns per-window inverse-
#     frequency weights that balance the three regime classes (clear / mixed /
#     overcast) across the whole concatenated dataset.  The weight of each window
#     is proportional to 1 / (frequency of its dominant regime across ALL windows
#     from ALL stations).  This is equivalent to importance-sampling toward a
#     uniform regime distribution without discarding any data.
# 
#     Pass the weights to a WeightedRandomSampler in the DataLoader:
#         weights = train_ds.get_sampling_weights()
#         sampler = WeightedRandomSampler(weights, num_samples=len(weights))
#         DataLoader(train_ds, sampler=sampler, ...)
# 
#     Val and test datasets still use uniform weights (identity) so that
#     metrics reflect the true climate distribution at each site.
# 
# - Mild augmentation (training only):
#     Gaussian noise on sunlit K* and temporal jitter. No cloud injection.
#     Real multi-station data provides all the regime diversity needed.
#     Night values are never touched.
# 
# - Observation channels (VAE encoder only):
#     Rich per-timestep measurements (DNI, diffuse fraction, ramp events, etc.)
#     are fed to the VAE encoder during training, enriching the latent space
#     with detailed cloud morphology information.  They are NEVER used at
#     generation time - the encoder is not called during generation; only
#     the decoder is.
# 
# Public API
# ----------
# build_datasets(train_profiles, val_profiles, test_profiles, cfg,
#                intraday_stats, day_feat_stats, obs_stats=None)
#     -> train_ds, val_ds, test_ds
# 
# collate_fn(batch)
#     -> dict of padded tensors
# 
# Each __getitem__ returns a dict with keys:
#     k_star_list   list[W] of (T_sun_i,)      float32  - sunlit K* per day
#     intraday_list list[W] of (T_sun_i, 3)   float32  - normalised physics
#     obs_list      list[W] of (T_sun_i, N_obs) float32 - obs channels (encoder only)
#     day_features  (W, 7)   float32  - normalised day-level solar geometry
#     location      (4,)     float32  - sin/cos lat/lon encoding
#     regime        (W,)     int64    - regime label (for regime_loss_weights_diff only)
#     T_sun         (W,)     int64    - actual sunlit lengths
#     date          list[str]
#     station       str
#

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.physics_utils import (
    DayFeatureNormStats,
    IntraDayNormStats,
    ObsPhysNormStats,
    ClimateFeatNormStats,
    extract_climate_features,
    extract_day_features,
    extract_intraday_matrix,
    extract_location_features,
    extract_obs_matrix,
    extract_sunlit_mask,
    N_CLIMATE_FEATURES,
)

log = logging.getLogger(__name__)


# -- Regime label constants ----------------------------------------------------
# Used ONLY for regime_loss_weights_diff in the diffusion training loss (train.py).
# NOT used as a conditioning signal for the denoiser.
# Must match cfg["data"]["regime"]["n_components"] = 4 and the ascending-tau ordering
# (clear=0, mixed_clear=1, mixed_overcast=2, overcast=3).

REGIME_CLEAR         = 0
REGIME_MIXED_CLEAR   = 1
REGIME_MIXED_OVERCAST = 2
REGIME_OVERCAST      = 3

# Backward-compat alias - old 3-class code used REGIME_MIXED=1 for the single mixed class.
# The 4-class model splits this into MIXED_CLEAR and MIXED_OVERCAST.
REGIME_MIXED = REGIME_MIXED_CLEAR   # kept so any remaining 3-class callers don't crash


# -- P1: RegimeGMM - data-driven regime labelling ------------------------------
# Fits an N-component GMM on (mean_k, std_k) per sunlit day.
# N_COMPONENTS is read from cfg["data"]["regime"]["n_components"] (default 4).
# Components are auto-labelled in ascending tau order:
#   highest mean_k -> 0 (clear)  ...  lowest mean_k -> N-1 (overcast).

class RegimeGMM:
    #N-component GMM on (mean_k, std_k) for regime classification.
    # 
    # Replaces all hard-threshold logic (clear_threshold / overcast_threshold /
    # variability_threshold).  The GMM learns the boundary geometry from data;
    # thresholds are no longer magic numbers.
    # 
    # N_COMPONENTS is read from cfg["data"]["regime"]["n_components"] at fit()
    # time (default 4).  The Markov matrix is NxN.
    # 
    # Usage
    # -----
    # gmm = RegimeGMM().fit(train_profiles, zenith_threshold_deg, cfg=cfg)
    # regime_int = gmm.label_day(mean_k, std_k, frac_below)   # -> 0 ... N-1
    # gmm.save("cache/regime_gmm.json")
    # 
    # gmm2 = RegimeGMM().load("cache/regime_gmm.json")
    # 
    # Attributes (after fit/load)
    # ---------------------------
    # n_components         : int - number of GMM components (from config)
    # gmm                  : sklearn GaussianMixture
    # component_to_regime  : ndarray (n_components,) int  - component -> regime label
    # markov_transition    : ndarray (n_components, n_components) float - row-stochastic
    #

    def __init__(self) -> None:
        self.n_components: int                  = 4   # overridden by fit() / load()
        self.gmm                                = None
        self.component_to_regime: Optional[np.ndarray] = None
        self.markov_transition:   Optional[np.ndarray] = None
        self._fitted = False

    # -- fit ------------------------------------------------------------------

    def fit(
        self,
        profiles: List[Dict],
        zenith_threshold: float,
        random_state: int = 42,
        cfg: Optional[Dict] = None,
    ) -> "RegimeGMM":
        #Fit on training profiles.  Returns self for chaining.
        # 
        # cfg is used to read n_components, n_init, and covariance_type from
        # cfg["data"]["regime"].  Falls back to defaults (4, 10, "full") when None.
        #
        from sklearn.mixture import GaussianMixture

        # Read n_components from config so it's never hardcoded
        if cfg is not None:
            rc = cfg.get("data", {}).get("regime", {})
            n_comp       = int(rc.get("n_components",    4))
            n_init       = int(rc.get("n_init",         10))
            cov_type     = str(rc.get("covariance_type", "full"))
        else:
            n_comp, n_init, cov_type = 4, 10, "full"

        self.n_components = n_comp

        feats = self._extract_features(profiles, zenith_threshold)
        if len(feats) < self.n_components * 5:
            raise ValueError(
                f"RegimeGMM.fit: only {len(feats)} usable days "
                f"(need >={self.n_components * 5}). Check training profiles."
            )

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=cov_type,
            n_init=n_init,
            max_iter=300,
            random_state=random_state,
        )
        gmm.fit(feats)
        self.gmm = gmm

        # Auto-label components in descending mean_k order:
        # highest mean_k -> regime 0 (clear), lowest -> regime n_components-1 (overcast).
        mean_k_per_comp = gmm.means_[:, 0]                  # (n_components,)
        order = np.argsort(mean_k_per_comp)[::-1]           # [highest ... lowest]
        self.component_to_regime = np.empty(self.n_components, dtype=np.int64)
        for rank, comp_idx in enumerate(order):
            self.component_to_regime[comp_idx] = rank

        # Log component stats
        comp_log = "  ".join(
            f"C{i}=(mean_k={gmm.means_[i,0]:.3f},std_k={gmm.means_[i,1]:.3f})->r{self.component_to_regime[i]}"
            for i in range(self.n_components)
        )
        log.info("RegimeGMM fitted | n_days=%d | n_components=%d | %s",
                 len(feats), self.n_components, comp_log)

        # All GMM state is now set - mark fitted before computing Markov labels
        self._fitted = True

        # Markov transition matrix on the labelled sequence (nxn)
        # feats columns: [mean_k, std_k, frac_below] - pass all three to label_day
        labels = [self.label_day(f[0], f[1], f[2] if len(f) > 2 else 0.0) for f in feats]
        self.markov_transition = self._markov(labels)
        n = self.n_components
        log.info(
            "Markov transition (%dx%d):\n%s",
            n, n,
            "\n".join(
                "  r%d->[%s]" % (r, " ".join("%.3f" % self.markov_transition[r, c] for c in range(n)))
                for r in range(n)
            ),
        )
        return self

    # -- label helpers ---------------------------------------------------------

    def label_day(self, mean_k: float, std_k: float, frac_below: float = 0.0) -> int:
        #Return REGIME_* int for one day.
        # 
        # frac_below: fraction of sunlit timesteps where K* < 0.3.
        # Required for the 3-feature GMM fitted by the current _extract_features().
        # Defaults to 0.0 for backward-compatibility when called from legacy code
        # that passes only (mean_k, std_k) - the result will be less accurate for
        # mixed-regime days but will not crash.
        #
        self._check_fitted()
        n_feat = self.gmm.means_.shape[1]
        if n_feat == 3:
            feat = np.array([[mean_k, std_k, frac_below]], dtype=np.float64)
        else:
            # Legacy 2-feature GMM loaded from an old cache file
            feat = np.array([[mean_k, std_k]], dtype=np.float64)
        comp = int(self.gmm.predict(feat)[0])
        return int(self.component_to_regime[comp])

    def label_days_batch(self, mean_k: np.ndarray, std_k: np.ndarray,
                         frac_below: Optional[np.ndarray] = None) -> np.ndarray:
        #Vectorised: (N,) arrays -> (N,) int REGIME_* labels.
        # 
        # frac_below: fraction of sunlit timesteps where K* < 0.3 per day.
        # Required when GMM was fitted with 3 features. Defaults to zeros
        # (safe fallback for legacy callers with 2-feature GMMs).
        #
        self._check_fitted()
        n_feat = self.gmm.means_.shape[1]
        if n_feat == 3:
            if frac_below is None:
                frac_below = np.zeros_like(mean_k)
            feats = np.stack([mean_k, std_k, frac_below], axis=1).astype(np.float64)
        else:
            feats = np.stack([mean_k, std_k], axis=1).astype(np.float64)
        comps = self.gmm.predict(feats)
        return self.component_to_regime[comps]

    # -- persist ---------------------------------------------------------------

    def save(self, path: str) -> None:
        #Serialise to JSON (no pickle - reproducible across sklearn versions).
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components":        self.n_components,
            "means":               self.gmm.means_.tolist(),
            "covariances":         self.gmm.covariances_.tolist(),
            "weights":             self.gmm.weights_.tolist(),
            "precisions_chol":     (self.gmm.precisions_cholesky_
                                    if hasattr(self.gmm, "precisions_cholesky_")
                                    else self.gmm.precisions_chol_).tolist(),
            "component_to_regime": self.component_to_regime.tolist(),
            "markov_transition":   (self.markov_transition.tolist()
                                    if self.markov_transition is not None else None),
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("RegimeGMM saved -> %s  (n_components=%d)", path, self.n_components)

    def load(self, path: str) -> "RegimeGMM":
        #Load from JSON.  Returns self for chaining.
        from sklearn.mixture import GaussianMixture
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"RegimeGMM file not found: {path.resolve()}.  "
                "Run build_datasets() on training data first."
            )
        with path.open() as f:
            blob = json.load(f)

        nc = int(blob["n_components"])
        self.n_components = nc

        gmm = GaussianMixture(n_components=nc, covariance_type="full")
        gmm.means_           = np.array(blob["means"],           dtype=np.float64)
        gmm.covariances_     = np.array(blob["covariances"],     dtype=np.float64)
        gmm.weights_         = np.array(blob["weights"],         dtype=np.float64)
        gmm.precisions_cholesky_ = np.array(blob["precisions_chol"], dtype=np.float64)
        gmm.precisions_chol_     = gmm.precisions_cholesky_   # backward-compat alias
        # Reconstruct precisions_ from Cholesky so predict() works without calling fit()
        prec_chol = gmm.precisions_cholesky_
        gmm.precisions_ = np.array(
            [prec_chol[k] @ prec_chol[k].T for k in range(nc)],
            dtype=np.float64,
        )
        gmm.converged_ = True
        gmm.n_iter_    = 0
        self.gmm = gmm

        self.component_to_regime = np.array(blob["component_to_regime"], dtype=np.int64)
        mt = blob.get("markov_transition")
        self.markov_transition = np.array(mt, dtype=np.float64) if mt is not None else None
        self._fitted = True
        log.info("RegimeGMM loaded <- %s  (n_components=%d)", path, self.n_components)
        return self

    # -- private ---------------------------------------------------------------

    @staticmethod
    def _extract_features(profiles: List[Dict], zenith_threshold: float) -> np.ndarray:
        rows = []
        for p in profiles:
            k_arr = np.array(p["csi"], dtype=np.float32)
            mask  = extract_sunlit_mask(p, zenith_threshold)
            k_sun = k_arr[mask]
            if len(k_sun) < 2:
                continue
            mean_k     = float(np.mean(k_sun))
            std_k      = float(np.std(k_sun))
            # 3rd feature: fraction of sunlit time deeply overcast (K* < 0.3).
            # At 10-min resolution the clearness-index distribution is near-bimodal,
            # so (mean_k, std_k) alone cannot separate mixed_clear from mixed_overcast
            # - both sit in the sparse bridge region with similar mean and std.
            # frac_below_0.3 is orthogonal to that 2-D degeneracy:
            #   mixed_clear    -> few timesteps below 0.3  (low frac_below)
            #   mixed_overcast -> many timesteps below 0.3 (high frac_below)
            # This breaks the boundary instability that caused label swapping
            # between runs and stations.
            frac_below = float(np.mean(k_sun < 0.3))
            rows.append([mean_k, std_k, frac_below])
        return np.array(rows, dtype=np.float64)

    def _markov(self, labels: List[int]) -> np.ndarray:
        #Row-stochastic nxn transition matrix with Laplace smoothing.
        n = self.n_components
        T = np.ones((n, n), dtype=np.float64)   # 1-count Laplace prior
        for a, b in zip(labels[:-1], labels[1:]):
            if 0 <= a < n and 0 <= b < n:
                T[a, b] += 1.0
        return T / T.sum(axis=1, keepdims=True)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RegimeGMM is not fitted. Call fit() or load() first.")


# -- Station lat/lon fallback table --------------------------------------------
# Used when a profile dict is missing "lat"/"lon" keys - e.g. because an older
# version of SolarPreprocessor did not write them into the profile JSON.
#
# Keys are lower-cased station names exactly as stored in profile["station"].
# Add new stations here when needed; this is the single source of truth for
# location when the profile JSON does not carry it.
#
# All coordinates: (latitude_deg, longitude_deg), WGS-84.
STATION_LATLON: Dict[str, Tuple[float, float]] = {
    # SURFRAD network (https://gml.noaa.gov/grad/surfrad/sitepage.html)
    "bon":  ( 40.0519,  -88.3731),   # Bondville, IL
    "dra":  ( 36.6231, -116.0195),   # Desert Rock, NV
    "fpk":  ( 48.3077, -105.1017),   # Fort Peck, MT
    "gwn":  ( 34.2547,  -89.8729),   # Goodwin Creek, MS
    "psu":  ( 40.7200,  -77.9309),   # Penn State, PA
    "sxf":  ( 43.7335,  -96.6233),   # Sioux Falls, SD
    "tbl":  ( 40.1249, -105.2368),   # Table Mountain, CO
    # ARM network
    "sgp":  ( 36.6050,  -97.4850),   # Southern Great Plains, OK
    "nsa":  ( 71.3230, -156.6150),   # North Slope Alaska
    "twp":  ( -2.0600,  147.4250),   # Tropical Western Pacific
}


def _label_regime_gmm(k_star_sunlit: np.ndarray, gmm: "RegimeGMM") -> int:
    #Label one day using a fitted RegimeGMM.  Falls back to REGIME_CLEAR for empty arrays.
    # 
    # Computes all 3 features (mean_k, std_k, frac_below_0.3) and passes them to
    # label_day().  label_day() handles legacy 2-feature GMMs transparently.
    #
    if len(k_star_sunlit) < 2:
        return REGIME_CLEAR
    mean_k     = float(np.mean(k_star_sunlit))
    std_k      = float(np.std(k_star_sunlit))
    frac_below = float(np.mean(k_star_sunlit < 0.3))
    return gmm.label_day(mean_k, std_k, frac_below)


# -- Core dataset --------------------------------------------------------------

class SolarSequenceDataset(Dataset):
    #Sliding-window dataset over sorted daily K* profiles.
    # 
    # Parameters
    # ----------
    # profiles       : list of dicts from SolarPreprocessor, sorted by date,
    #                  all from the SAME station.  Each dict must contain lat/lon.
    # cfg            : full config dict
    # intraday_stats : fitted IntraDayNormStats
    # day_feat_stats : fitted DayFeatureNormStats
    # obs_stats      : fitted ObsPhysNormStats | None
    # regime_gmm     : fitted RegimeGMM - replaces hard-threshold classification.
    #                  Pass None only for empty datasets (e.g. empty test split).
    # split          : 'train' | 'val' | 'test'
    #

    def __init__(
        self,
        profiles: List[Dict],
        cfg: Dict,
        intraday_stats: IntraDayNormStats,
        day_feat_stats: DayFeatureNormStats,
        obs_stats: Optional[ObsPhysNormStats] = None,
        regime_gmm: Optional["RegimeGMM"] = None,
        split: str = "train",
        relabel_map: Optional[Dict[str, int]] = None,
        climate_stats: Optional[ClimateFeatNormStats] = None,
    ):
        super().__init__()
        self.cfg            = cfg
        self.intraday_stats = intraday_stats
        self.day_feat_stats = day_feat_stats
        self.obs_stats      = obs_stats
        self.climate_stats  = climate_stats   # None = climate features disabled
        self.regime_gmm     = regime_gmm
        self.split          = split
        self.is_train       = split == "train"

        # relabel_map: optional dict mapping profile date-string (YYYY-MM-DD) to
        # k-means regime label, overriding the GMM label during Stage-2 training.
        # Built from cache/latent_relabels.npy + the training profile ordering.
        # Only applied when use_latent_relabels=True in cfg["data"]["regime"] and
        # relabel_map is not None.
        self._relabel_map: Optional[Dict[str, int]] = relabel_map

        d_cfg = cfg["data"]
        self.W            = d_cfg["window_size"]
        self.stride       = d_cfg.get("window_stride", 1)
        self.zenith_thr   = d_cfg["zenith_threshold_deg"]
        # P2: do NOT read clear_threshold / overcast_threshold / variability_threshold
        self.noise_std    = d_cfg["augmentation"]["noise_std"]
        self.jitter       = d_cfg["augmentation"]["temporal_jitter"]
        self.k_max_clip   = d_cfg["augmentation"].get("k_max_clip", cfg["physics"]["k_max"])
        self.max_gap_days = d_cfg.get("max_gap_days", 1)

        # Observation channels fed to the VAE encoder during training only.
        self.obs_channels: List[str] = cfg["vae"].get("obs_channels", [])

        # Sort profiles by date - mandatory for sequential windows
        self.profiles = sorted(profiles, key=lambda p: p["date"])

        # Build windows and pre-compute regime labels for loss weighting
        self._windows:        List[int] = self._build_windows()
        self._window_regimes: List[int] = self._label_windows()

        log.info(
            "SolarSequenceDataset [%s] | station=%s | profiles=%d | windows=%d | W=%d",
            split,
            profiles[0].get("station", "?") if profiles else "?",
            len(self.profiles),
            len(self._windows),
            self.W,
        )

    # -- Window construction ---------------------------------------------------

    def _build_windows(self) -> List[int]:
        #Return list of valid window start indices.
        # 
        # A window is valid only if all W consecutive profiles are temporally
        # contiguous.  max_gap_days=1 means no missing calendar day is allowed.
        #
        n       = len(self.profiles)
        windows = []
        for start in range(0, n - self.W + 1, self.stride):
            contiguous = True
            for i in range(start, start + self.W - 1):
                d0    = self.profiles[i]["date"]
                d1    = self.profiles[i + 1]["date"]
                delta = (np.datetime64(d1, "D") - np.datetime64(d0, "D")).astype(int)
                if delta > self.max_gap_days:
                    contiguous = False
                    break
            if contiguous:
                windows.append(start)
        return windows

    def _label_windows(self) -> List[int]:
        #Label each window by its dominant day-regime (for loss weighting only).
        # 
        # Uses RegimeGMM when available.  Falls back to mean_k > 0.5 binary split
        # when regime_gmm is None (empty-split construction only).
        #
        labels = []
        for start in self._windows:
            counts: Dict[int, int] = defaultdict(int)
            for i in range(start, start + self.W):
                p     = self.profiles[i]
                k_arr = np.array(p["csi"], dtype=np.float32)
                mask  = extract_sunlit_mask(p, self.zenith_thr)
                k_sun = k_arr[mask]
                if self.regime_gmm is not None:
                    r = _label_regime_gmm(k_sun, self.regime_gmm)
                else:
                    # Fallback for empty splits - no GMM available.
                    # Use REGIME_OVERCAST (last class) for low-K* days.
                    r = REGIME_CLEAR if (len(k_sun) > 0 and float(np.mean(k_sun)) > 0.5) \
                        else REGIME_OVERCAST
                counts[r] += 1
            dominant = max(counts, key=lambda r: (counts[r], r))
            labels.append(dominant)
        return labels

    # -- Mild augmentation -----------------------------------------------------

    def _augment_sunlit_k(
        self,
        k:        np.ndarray,   # (T_sun,)        sunlit K*
        phys:     np.ndarray,   # (T_sun, 3)      normalised intraday physics
        obs_norm: np.ndarray,   # (T_sun, N_obs)  normalised obs channels
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Gaussian noise + temporal jitter on sunlit K* (training only).
        # 
        # k, phys, and obs_norm are all shifted by the same jitter offset so
        # that timestep i always aligns across all three arrays.
        # 
        # Jitter implementation - trim, never pad
        # ----------------------------------------
        # The old implementation filled the vacated boundary positions with
        # zeros for K* and repeated boundary rows for physics.  Both are
        # unprincipled:
        # 
        #   - K*=0 at the edge of the sunlit window is physically "night" /
        #     "total overcast".  Injecting it into a clear day's encoder input
        #     drags the mean K* down, biasing the latent representation toward
        #     a cloudier regime that does not exist in the real data.
        # 
        #   - Repeating the first/last physics row is a constant extrapolation
        #     across a sunrise/sunset boundary where zenith is changing rapidly,
        #     producing mismatched (physics, K*) pairs with no physical analogue.
        # 
        # Correct approach: after shifting by |shift| timesteps, simply drop
        # the |shift| boundary positions that have no real counterpart.  The
        # returned arrays are |shift| timesteps shorter.  The caller appends
        # len(k_sun) to t_sun_list, which becomes T_sun in the batch; collate_fn
        # pads to T_max with zeros and valid_mask=False, so the encoder ignores
        # the vacated positions entirely.  No fake values ever enter the
        # distribution.
        #
        import random as _random
        assert k.shape[0] == phys.shape[0] == obs_norm.shape[0]

        # Gaussian noise on K* only - physics are deterministic
        k = k + np.random.normal(0.0, self.noise_std, size=k.shape).astype(np.float32)
        k = np.clip(k, 0.0, self.k_max_clip)

        # Temporal jitter: trim boundary positions instead of padding with fake data.
        # shift > 0 -> window moves right (drop first |shift| timesteps).
        # shift < 0 -> window moves left  (drop last  |shift| timesteps).
        # Guard: never trim so much that the array becomes empty.
        if self.jitter > 0:
            shift = _random.randint(-self.jitter, self.jitter)
            if shift != 0:
                abs_shift = abs(shift)
                if abs_shift < len(k):
                    if shift > 0:
                        k        = k[abs_shift:]
                        phys     = phys[abs_shift:]
                        obs_norm = obs_norm[abs_shift:]
                    else:
                        k        = k[:-abs_shift]
                        phys     = phys[:-abs_shift]
                        obs_norm = obs_norm[:-abs_shift]
                # If abs_shift >= len(k) the array would become empty - skip jitter.

        assert k.shape[0] == phys.shape[0] == obs_norm.shape[0], \
            "post-jitter length mismatch"
        return k, phys, obs_norm

    # -- __getitem__ -----------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start = self._windows[idx]

        k_star_list:   List[np.ndarray] = []
        intraday_list: List[np.ndarray] = []
        obs_list:      List[np.ndarray] = []
        day_feat_list: List[np.ndarray] = []
        regime_list:   List[int]        = []
        t_sun_list:    List[int]        = []
        date_list:     List[str]        = []

        for i in range(start, start + self.W):
            p    = self.profiles[i]
            mask = extract_sunlit_mask(p, self.zenith_thr)

            k_sun     = np.array(p["csi"], dtype=np.float32)[mask]
            # Hard-clip raw K* to [0, k_max_clip] before augmentation.
            # k_max_clip=1.30 (from config) preserves cloud-edge enhancement
            # events (K* 1.05-1.30) which are real physics at SURFRAD stations.
            # This clip only removes pathological values beyond p99=1.30 that
            # would produce extreme FFT losses without physical meaning.
            k_sun = np.clip(k_sun, 0.0, float(
                self.cfg["data"]["augmentation"].get("k_max_clip",
                self.cfg["physics"]["k_max"])
            ))
            phys_norm = self.intraday_stats.normalize(extract_intraday_matrix(p)[mask])
            day_feat  = self.day_feat_stats.normalize(
                extract_day_features(p, self.zenith_thr)
            )

            # Climate features: append to day_feat so the denoiser sees
            # climatological context at every day-token.
            # Computed once per unique (lat, lon) - kgcpy lookup is fast (~1ms).
            # At inference the same call is made in _compute_solar_geometry.
            # Produces N_CLIMATE_FEATURES=10 dims: 6 Koppen one-hot + 4 irr quantiles.
            # day_feat goes from 7-dim (solar) to 17-dim (solar + climate).
            if self.climate_stats is not None:
                lat = float(p.get("lat", 0.0))
                lon = float(p.get("lon", 0.0))
                clim_raw  = extract_climate_features(lat, lon)       # (10,)
                clim_norm = self.climate_stats.normalize(clim_raw)   # (10,)
                day_feat  = np.concatenate([day_feat, clim_norm])    # (7+10=17,)

            # Observation channels - encoder only; empty array when not configured.
            if self.obs_channels:
                obs_raw = extract_obs_matrix(p, self.obs_channels, self.zenith_thr)
            else:
                obs_raw = np.empty((int(mask.sum()), 0), dtype=np.float32)

            # Normalise obs channels.
            # Binary event flags ({0,1}) must NOT be z-scored - they produce
            # 3-33-sigma outliers that destabilise encoder training.
            # obs_stats.binary_mask marks which columns to leave as {0,1}.
            if self.obs_stats is not None and obs_raw.shape[1] > 0:
                b_mask = getattr(self.obs_stats, "binary_mask", None)
                if b_mask is not None and b_mask.any():
                    obs_norm = obs_raw.copy()
                    continuous = ~b_mask
                    if continuous.any():
                        obs_norm[:, continuous] = self.obs_stats.normalize(
                            obs_raw[:, continuous]
                        )
                        obs_norm[:, continuous] = np.clip(
                            obs_norm[:, continuous], -3.0, 3.0
                        )
                    # binary columns pass through as {0,1}
                else:
                    obs_norm = self.obs_stats.normalize(obs_raw)
            else:
                obs_norm = obs_raw

            # Note 6 FIX: compute regime label from RAW k_sun BEFORE augmentation.
            # The GMM was fitted on raw (un-augmented) K* values via _extract_features.
            # Computing the label after _augment_sunlit_k means jitter-trimmed or
            # noise-shifted K* affects the label - semantically wrong and inconsistent
            # with GMM fitting. The relabel_map override (Stage 2) is unaffected.
            if self.regime_gmm is not None:
                regime = _label_regime_gmm(k_sun, self.regime_gmm)
            else:
                regime = REGIME_CLEAR if (len(k_sun) > 0 and float(np.mean(k_sun)) > 0.5) \
                         else REGIME_OVERCAST

            # Mild noise + jitter (training only).
            # Pass obs_norm into _augment_sunlit_k so all three arrays are
            # trimmed together by the same shift - no alignment heuristics needed.
            if self.is_train and len(k_sun) > 0:
                k_sun, phys_norm, obs_norm = self._augment_sunlit_k(
                    k_sun, phys_norm, obs_norm
                )

            # Stage-2 relabel map override: replace GMM/raw label with the
            # k-means latent-space aligned label for this day.
            if self._relabel_map is not None:
                day_str = p.get("date", None)
                if day_str is not None and day_str in self._relabel_map:
                    regime = int(self._relabel_map[day_str])

            k_star_list.append(k_sun)
            intraday_list.append(phys_norm)
            obs_list.append(obs_norm)
            day_feat_list.append(day_feat)
            regime_list.append(regime)
            t_sun_list.append(len(k_sun))
            date_list.append(p["date"])

        # Location encoding from profile lat/lon.
        # Primary source: profile["lat"] / profile["lon"] written by SolarPreprocessor.
        # Fallback:       STATION_LATLON table keyed by lower-cased profile["station"].
        # Error:          both missing -> raises so the user is never silently at (0, 0).
        p0      = self.profiles[start]
        lat_raw = p0.get("lat", None)
        lon_raw = p0.get("lon", None)
        if lat_raw is not None and lon_raw is not None:
            lat = float(lat_raw)
            lon = float(lon_raw)
        else:
            st_key = str(p0.get("station", "")).lower().strip()
            if st_key in STATION_LATLON:
                lat, lon = STATION_LATLON[st_key]
                log.debug(
                    "Profile for station '%s' has no lat/lon - "
                    "using table fallback (%.4f, %.4f).",
                    st_key, lat, lon,
                )
            else:
                raise ValueError(
                    f"Profile for station {p0.get('station', '?')!r} has no lat/lon "
                    f"and is not in STATION_LATLON.  "
                    f"Either reprocess the data (add lat/lon to each profile dict) "
                    f"or add the station to STATION_LATLON in dataset.py."
                )
        location = extract_location_features(lat, lon)   # (4,)

        return {
            "k_star_list":   k_star_list,
            "intraday_list": intraday_list,
            "obs_list":      obs_list,
            "day_features":  torch.from_numpy(np.stack(day_feat_list)),   # (W, 7)
            "location":      torch.from_numpy(location),                   # (4,)
            "lat":           lat,
            "lon":           lon,
            # regime is for regime_loss_weights_diff in train.py ONLY, not a model input
            "regime":        torch.tensor(regime_list, dtype=torch.long),  # (W,)
            "T_sun":         torch.tensor(t_sun_list,  dtype=torch.long),  # (W,)
            "date":          date_list,
            "station":       p0.get("station", ""),
        }

    def get_sampling_weights(self) -> List[float]:
        #Per-window inverse-regime-frequency weights for WeightedRandomSampler.
        # 
        # Previous (broken) approach
        # ---------------------------
        # Each window was assigned a single dominant-regime label (plurality vote
        # across its W days) and weighted by 1 / count(that label).  With W=14
        # and a typical clear-sky station (>60% clear), almost every window was
        # labelled CLEAR regardless of how many mixed or overcast days it contained.
        # A window with 12 clear + 2 overcast days received the same low weight as
        # a window with 14 clear days - the 2 overcast days inside it were never
        # upweighted.  The sampler appeared balanced at the window level but the
        # individual cloudy days were systematically under-represented.
        # 
        # Correct approach - mean per-day weight
        # ----------------------------------------
        # 1. Compute the per-day inverse-frequency weight for every individual day
        #    across all windows in this station's dataset:
        #        day_weight[date] = 1 / count(regime of that date across all days
        #                                     seen in any window of this dataset)
        #    This is the natural importance weight for a day: rare-regime days
        #    receive a high weight, common-regime days receive a low weight.
        # 
        # 2. A window's weight is the MEAN of the per-day weights of its W
        #    constituent days.  A window that contains 12 clear + 2 overcast days
        #    gets a substantially higher weight than an all-clear window because
        #    the 2 overcast days each contribute their high individual weight to
        #    the mean.  The model therefore sees the cloudy days inside mostly-
        #    clear sequences - which is exactly the climatologically correct context
        #    in which they appear.
        # 
        # 3. All days are counted once per occurrence in a window (overlapping
        #    windows mean a day appears in up to W windows; all occurrences count).
        #    This is consistent with how WeightedRandomSampler uses the weights:
        #    it samples proportionally, so a window with high mean weight is drawn
        #    more often and its rare-regime days are seen more often.
        # 
        # CRITICAL FIX: when _relabel_map is active (Stage-2 training), regime
        # labels are sourced from the k-means latent-space map - NOT from the GMM.
        # Previously the sampler always used GMM labels while the loss used
        # k-means labels, creating a silent mismatch: windows the sampler
        # considered "clear" could be "overcast" in the loss, so overcast days
        # were under-sampled despite the loss trying to upweight them.
        #
        if not self._windows:
            return []

        # Step 1: count per-regime frequency across all individual days in all windows.
        # A day that appears in multiple overlapping windows is counted each time.
        regime_day_counts: Dict[int, int] = defaultdict(int)
        all_day_regimes: List[List[int]] = []   # parallel to self._windows

        for start in self._windows:
            window_day_regimes: List[int] = []
            for i in range(start, start + self.W):
                p     = self.profiles[i]
                k_arr = np.array(p["csi"], dtype=np.float32)
                mask  = extract_sunlit_mask(p, self.zenith_thr)
                k_sun = k_arr[mask]

                # CRITICAL FIX: respect relabel_map when active so that the
                # sampler and the training loss use the same regime label
                # for every day.  Stage-2 relabels (k-means latent geometry)
                # can disagree with GMM labels for mixed-regime days.
                day_str = p.get("date", None)
                if self._relabel_map is not None and day_str in self._relabel_map:
                    r = int(self._relabel_map[day_str])
                elif self.regime_gmm is not None:
                    r = _label_regime_gmm(k_sun, self.regime_gmm)
                else:
                    r = REGIME_CLEAR if (len(k_sun) > 0 and float(np.mean(k_sun)) > 0.5) \
                        else REGIME_OVERCAST

                window_day_regimes.append(r)
                regime_day_counts[r] += 1
            all_day_regimes.append(window_day_regimes)

        # Step 2: each window's weight = mean of per-day inverse-frequency weights.
        weights: List[float] = []
        for window_day_regimes in all_day_regimes:
            day_weights = [
                1.0 / max(regime_day_counts[r], 1)
                for r in window_day_regimes
            ]
            weights.append(float(np.mean(day_weights)))

        return weights

    def __len__(self) -> int:
        return len(self._windows)


# -- Collate function ----------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    #Pad variable-length sunlit slices to the longest in the batch.
    # 
    # Returns
    # -------
    # k_star        : (B, W, T_max)         float32
    # valid_mask    : (B, W, T_max)         bool     - True = real timestep
    # intraday_phys : (B, W, T_max, 3)     float32  - normalised deterministic physics
    # obs_phys      : (B, W, T_max, N_obs) float32  - encoder-only obs channels
    #                 N_obs=0 when obs_channels is empty.
    # day_features  : (B, W, 7)            float32
    # location      : (B, 4)               float32
    # regime        : (B, W)               int64    - for regime_loss_weights_diff only
    # T_sun         : (B, W)               int64
    # date          : list[list[str]]
    # station       : list[str]
    # lat_lon       : (B, 2)               float32  - raw degrees
    #
    B = len(batch)
    W = batch[0]["day_features"].shape[0]

    T_max = max(int(t) for s in batch for t in s["T_sun"])
    T_max = max(T_max, 1)   # guard: polar night edge case

    # Infer N_obs from first non-empty obs slice
    N_obs = 0
    for s in batch:
        for arr in s["obs_list"]:
            if arr.ndim == 2:
                N_obs = arr.shape[1]
                break
        if N_obs > 0:
            break
    for s in batch:
        for arr in s["obs_list"]:
            assert arr.ndim == 2
            assert arr.shape[1] == N_obs, (
                f"obs channel mismatch: expected {N_obs}, got {arr.shape[1]}"
            )

    k_star_out     = torch.zeros(B, W, T_max,        dtype=torch.float32)
    valid_mask_out = torch.zeros(B, W, T_max,        dtype=torch.bool)
    intraday_out   = torch.zeros(B, W, T_max, 3,     dtype=torch.float32)
    obs_out        = torch.zeros(B, W, T_max, N_obs, dtype=torch.float32)

    for b, s in enumerate(batch):
        for w in range(W):
            k  = torch.from_numpy(s["k_star_list"][w])
            ph = torch.from_numpy(s["intraday_list"][w])
            t  = k.shape[0]
            k_star_out[b, w, :t]      = k
            valid_mask_out[b, w, :t]  = True
            intraday_out[b, w, :t, :] = ph
            if N_obs > 0:
                ob = torch.from_numpy(s["obs_list"][w])
                obs_out[b, w, :t, :] = ob

    return {
        "k_star":        k_star_out,
        "valid_mask":    valid_mask_out,
        "intraday_phys": intraday_out,
        "obs_phys":      obs_out,           # (B, W, T_max, N_obs); N_obs=0 -> shape(...,0)
        "day_features":  torch.stack([s["day_features"] for s in batch]),   # (B, W, 7)
        "location":      torch.stack([s["location"]     for s in batch]),   # (B, 4)
        "regime":        torch.stack([s["regime"]       for s in batch]),   # (B, W)
        "T_sun":         torch.stack([s["T_sun"]        for s in batch]),   # (B, W)
        "date":          [s["date"]    for s in batch],
        "station":       [s["station"] for s in batch],
        "lat_lon":       torch.tensor(
                             [[s["lat"], s["lon"]] for s in batch],
                             dtype=torch.float32,
                         ),   # (B, 2)
    }


# -- Multi-station dataset builder ---------------------------------------------

class _MultiStationDataset(Dataset):
    #Concatenates per-station SolarSequenceDatasets into one iterable.
    # 
    # Windows from different stations are interleaved during iteration but
    # never mixed within a single window -- temporal contiguity is guaranteed
    # per-station.  Indexing is O(log n_stations) via searchsorted.
    # 
    # Sampling
    # --------
    # get_sampling_weights() returns per-window inverse-regime-frequency
    # weights computed over the GLOBAL regime distribution across all stations.
    # This corrects for between-station climate imbalance: a desert station
    # (high clear-sky fraction) and a humid station (high overcast fraction)
    # each contribute equally to every regime class after weighting.
    # 
    # Pass the returned weights to WeightedRandomSampler:
    #     sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    #     DataLoader(train_ds, sampler=sampler, batch_size=..., num_workers=...)
    #

    def __init__(
        self,
        sub_datasets: List[SolarSequenceDataset],
        cfg: Dict,
        intraday_stats: IntraDayNormStats,
        day_feat_stats: DayFeatureNormStats,
        split: str,
        regime_gmm: Optional["RegimeGMM"] = None,
    ):
        self._datasets      = sub_datasets
        self._cumlen        = np.cumsum([0] + [len(d) for d in sub_datasets])
        self.split          = split
        self.intraday_stats = intraday_stats
        self.day_feat_stats = day_feat_stats
        self.obs_stats      = sub_datasets[0].obs_stats if sub_datasets else None
        # P4: expose GMM and its Markov matrix so train.py can log them at startup
        self.regime_gmm        = regime_gmm
        self.markov_transition = (
            regime_gmm.markov_transition
            if regime_gmm is not None else None
        )

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx    = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
        local_idx = idx - int(self._cumlen[ds_idx])
        return self._datasets[ds_idx][local_idx]

    def get_sampling_weights(self) -> List[float]:
        #Per-window mean-per-day inverse-regime-frequency weights.
        # 
        # Delegates to each sub-dataset's get_sampling_weights() (which now
        # computes mean per-day weights rather than dominant-window labels),
        # then concatenates the results in the same order as __getitem__ indexing.
        # 
        # The per-station weighting is intentional: each station balances its own
        # regime frequency independently before concatenation.  This preserves the
        # true climatological ratio *between* stations (Desert Rock stays more
        # clear-persistent than Bondville) while still upweighting rare-regime days
        # within each station's windows.
        # 
        # Logging reports the effective mean weight by regime class so imbalance
        # can be monitored.
        #
        if not self._datasets:
            return []

        all_weights: List[float] = []
        for ds in self._datasets:
            station_weights = ds.get_sampling_weights()
            all_weights.extend(station_weights)

            station_name = (ds.profiles[0].get("station", "?")
                            if ds.profiles else "?")
            log.info(
                "Sampling weights [%s] | windows=%d  "
                "weight_mean=%.4f  weight_min=%.4f  weight_max=%.4f",
                station_name, len(station_weights),
                float(np.mean(station_weights)) if station_weights else 0.0,
                float(np.min(station_weights))  if station_weights else 0.0,
                float(np.max(station_weights))  if station_weights else 0.0,
            )

        total_w = max(sum(all_weights), 1e-8)
        log.info(
            "Sampling weights | GLOBAL total_windows=%d  "
            "effective weight share by regime (approx):",
            len(all_weights),
        )

        return all_weights


def recompute_markov_from_relabels(
    relabel_map:    Dict[str, int],
    train_profiles: List[Dict],
    n_components:   int,
    cfg:            Dict,
) -> np.ndarray:
    #Recompute the day-to-day Markov transition matrix from k-means relabels.
    # 
    # WHY THIS IS NEEDED
    # ------------------
    # RegimeGMM._markov() is computed once at fit() time from GMM-labelled day
    # sequences.  After Stage-1 ends, relabel_latents_kmeans() reassigns each
    # training day to a new regime label aligned with the learned latent geometry.
    # The GMM Markov matrix is now stale - mixed-regime transitions in particular
    # can differ substantially because k-means splits the (mean_k, std_k) space
    # differently from the learned tau-space.
    # 
    # The recomputed matrix is saved to
    #     cfg["paths"]["regime_gmm"] parent dir / "latent_markov.json"
    # and should be loaded by inference code (instead of regime_gmm.markov_transition)
    # when sampling day-to-day regime sequences at generation time.
    # 
    # Parameters
    # ----------
    # relabel_map    : {date_str -> k-means regime int} built by relabel_latents_kmeans
    # train_profiles : raw training profile list (sorted by station then date here)
    # n_components   : number of regime classes (must equal cfg["data"]["regime"]["n_components"])
    # cfg            : full config dict (used only for path lookup)
    # 
    # Returns
    # -------
    # markov : (n_components, n_components) float64 row-stochastic transition matrix
    #
    from collections import defaultdict as _dd
    from pathlib import Path as _Path
    import json as _json
    import numpy as _np

    n = n_components

    # Reconstruct sequential label sequences per station (preserve temporal order)
    by_station: Dict[str, List[int]] = _dd(list)
    for p in sorted(train_profiles, key=lambda x: (x.get("station", ""), x.get("date", ""))):
        day_str = p.get("date", None)
        if day_str is not None and day_str in relabel_map:
            by_station[p.get("station", "_unknown")].append(int(relabel_map[day_str]))

    # Laplace-smoothed (nxn) transition matrix - 1-count prior prevents zeros
    T = _np.ones((n, n), dtype=_np.float64)
    for station_labels in by_station.values():
        for a, b in zip(station_labels[:-1], station_labels[1:]):
            if 0 <= a < n and 0 <= b < n:
                T[a, b] += 1.0
    markov = T / T.sum(axis=1, keepdims=True)

    # Save alongside regime_gmm.json
    gmm_path    = _Path(cfg.get("paths", {}).get("regime_gmm", "cache/regime_gmm.json"))
    markov_path = gmm_path.parent / "latent_markov.json"
    markov_path.parent.mkdir(parents=True, exist_ok=True)
    with markov_path.open("w") as f:
        _json.dump({"markov_relabeled": markov.tolist(), "n_components": n}, f, indent=2)
    log.info(
        "Recomputed Markov matrix from k-means relabels -> %s\n"
        "  %s",
        markov_path,
        "\n  ".join(
            "r%d->[%s]" % (r, " ".join("%.3f" % markov[r, c] for c in range(n)))
            for r in range(n)
        ),
    )
    return markov


def build_datasets(
    train_profiles: List[Dict],
    val_profiles:   List[Dict],
    test_profiles:  List[Dict],
    cfg: Dict,
    intraday_stats: IntraDayNormStats,
    day_feat_stats: DayFeatureNormStats,
    obs_stats: Optional[ObsPhysNormStats] = None,
    regime_gmm: Optional["RegimeGMM"] = None,
    climate_stats: Optional[ClimateFeatNormStats] = None,
) -> Tuple["SolarSequenceDataset", "SolarSequenceDataset", "SolarSequenceDataset"]:
    #Build train/val/test datasets from pre-split profile lists.
    # 
    # P4 - GMM lifecycle
    # ------------------
    # If ``regime_gmm`` is None, a new GMM is fitted here on training profiles
    # and saved to ``cfg["paths"]["regime_gmm"]`` (default "cache/regime_gmm.json").
    # Pass a pre-fitted GMM (e.g. on resume) to skip refitting.
    # 
    # The same fitted GMM is forwarded to every split so val/test labelling is
    # consistent with training.
    # 
    # The returned ``train_ds.markov_transition`` is the (3x3) empirical
    # day->day regime transition matrix from the training sequence, which can be
    # used to drive regime sequences at generation time.
    # 
    # For multi-station training pass ``train_ds.get_sampling_weights()`` to a
    # WeightedRandomSampler; val/test should use plain sequential sampling.
    # 
    # Latent relabelling (Stage 2)
    # ----------------------------
    # When cfg["data"]["regime"]["use_latent_relabels"]=True and the file
    # cfg["paths"]["latent_relabels"] exists, the training dataset uses k-means
    # regime labels (aligned with the trained latent geometry) instead of frozen
    # GMM labels.  Val/test keep GMM labels so they remain interpretable.
    #
    zenith_thr = float(cfg["data"]["zenith_threshold_deg"])
    W          = int(cfg["data"]["window_size"])

    # -- P4: fit GMM once on training data ------------------------------------
    if regime_gmm is None:
        log.info("Fitting RegimeGMM on %d training profiles ...", len(train_profiles))
        regime_gmm = RegimeGMM().fit(train_profiles, zenith_thr, cfg=cfg)
        gmm_path = Path(cfg.get("paths", {}).get("regime_gmm", "cache/regime_gmm.json"))
        regime_gmm.save(str(gmm_path))
    else:
        log.info("Using pre-fitted RegimeGMM (skipping fit).")

    # -- Load latent relabel map for Stage-2 training --------------------------
    # Maps profile date-string -> k-means regime label.
    # Built by relabel_latents_kmeans() in train.py after Stage 1.
    # Applied to the training split only; val/test keep GMM labels.
    #
    # Loading strategy (preferred -> fallback):
    #
    # 1. JSON map  (cache/latent_relabels_map.json, written by the fixed
    #    relabel_latents_kmeans).  Format: {"YYYY-MM-DD": int, ...}.
    #    Order-independent; robust to any DataLoader shuffle.
    #
    # 2. Positional .npy  (cache/latent_relabels.npy, legacy).
    #    Only used when the JSON map is absent; emits an explicit WARNING
    #    because alignment with the training profile list is fragile and
    #    was the root cause of relabels being silently discarded (rule 2,
    #    previous session).  Users should re-run build_latent_cache to
    #    generate the JSON map.
    relabel_map: Optional[Dict[str, int]] = None
    use_relabels = cfg.get("data", {}).get("regime", {}).get("use_latent_relabels", True)

    if use_relabels:
        from collections import Counter as _Counter
        # -- preferred: JSON map --------------------------------------------
        json_map_path = Path(cfg.get("paths", {}).get(
            "latent_relabels_map", "cache/latent_relabels_map.json"
        ))
        npy_path = Path(cfg.get("paths", {}).get(
            "latent_relabels", "cache/latent_relabels.npy"
        ))

        if json_map_path.exists():
            try:
                with json_map_path.open() as _f:
                    _raw = json.load(_f)
                relabel_map = {str(k): int(v) for k, v in _raw.items()}
                log.info(
                    "Latent relabels loaded from JSON map %s - %d entries.  "
                    "Training regime labels will use k-means aligned labels.",
                    json_map_path, len(relabel_map),
                )
                new_counts = dict(sorted(_Counter(relabel_map.values()).items()))
                log.info("  k-means label distribution: %s", new_counts)
            except Exception as _e:
                log.warning(
                    "Failed to load latent relabels JSON map from %s: %s - "
                    "will try positional .npy fallback.",
                    json_map_path, _e,
                )
                relabel_map = None

        # -- fallback: positional .npy --------------------------------------
        if relabel_map is None and npy_path.exists():
            log.warning(
                "Latent relabels: JSON map not found at %s.  "
                "Falling back to positional .npy (%s).  "
                "This alignment is FRAGILE - re-run build_latent_cache to "
                "generate the JSON map and eliminate this risk.",
                json_map_path, npy_path,
            )
            try:
                relabels_arr = np.load(npy_path)
                # Align by sorting train_profiles by date - same ordering
                # assumed by the legacy build_latent_cache pass-0 iterator.
                sorted_train = sorted(train_profiles, key=lambda p: p["date"])
                if len(relabels_arr) == len(sorted_train):
                    relabel_map = {
                        str(p["date"]): int(relabels_arr[i])
                        for i, p in enumerate(sorted_train)
                    }
                    log.info(
                        "Latent relabels loaded from positional .npy %s - %d entries.",
                        npy_path, len(relabel_map),
                    )
                else:
                    log.warning(
                        "Positional .npy relabels length mismatch: "
                        "file has %d entries, training profiles has %d.  "
                        "Relabels ignored - re-run train_diffusion to rebuild cache.",
                        len(relabels_arr), len(sorted_train),
                    )
            except Exception as _e:
                log.warning(
                    "Failed to load positional .npy relabels from %s: %s - "
                    "using GMM labels.",
                    npy_path, _e,
                )

    def _make(profs: List[Dict], split: str, use_remap: bool = False) -> "SolarSequenceDataset":
        by_st: Dict[str, List[Dict]] = defaultdict(list)
        for p in profs:
            by_st[p["station"]].append(p)

        sub_datasets = [
            SolarSequenceDataset(
                st_profs, cfg, intraday_stats, day_feat_stats,
                obs_stats=obs_stats,
                regime_gmm=regime_gmm,
                split=split,
                relabel_map=relabel_map if use_remap else None,
                climate_stats=climate_stats,
            )
            for st_profs in by_st.values()
            if len(st_profs) >= W
        ]

        if not sub_datasets:
            log.warning("No profiles for split=%s - returning empty dataset", split)
            return SolarSequenceDataset(
                [], cfg, intraday_stats, day_feat_stats,
                obs_stats=obs_stats,
                regime_gmm=regime_gmm,
                split=split,
                relabel_map=None,
                climate_stats=climate_stats,
            )

        return _MultiStationDataset(
            sub_datasets, cfg, intraday_stats, day_feat_stats, split,
            regime_gmm=regime_gmm,
        )

    # Relabels applied to training split only; val/test use GMM labels for interpretability
    train_ds = _make(train_profiles, "train", use_remap=True)
    val_ds   = _make(val_profiles,   "val",   use_remap=False)
    test_ds  = _make(test_profiles,  "test",  use_remap=False)

    obs_ch = cfg["vae"].get("obs_channels", [])
    log.info(
        "Datasets built | train=%d  val=%d  test=%d windows | obs_channels=%s",
        len(train_ds), len(val_ds), len(test_ds), obs_ch or "none",
    )
    return train_ds, val_ds, test_ds