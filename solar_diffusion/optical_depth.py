#
# solar_diffusion/optical_depth.py
# ----------------------------------
# Optical-depth reparameterisation and noise schedule.

# Beer-Lambert: K*(t) = K_max · exp(−τ(t)), τ ≥ 0.
# Clear sky (K* = K_max) → τ = 0.  Total overcast (K* → 0) → τ → ∞.
# This maps the bounded K* domain onto the half-line where Gaussian noise is valid.

# Stage 2 diffuses AE latent codes z ∈ R^{d_z} in τ-space:
#   1. z   = vae.encode(k_star, physics, mask)
#   2. τ_0 = LatentTauTransform.to_tau_space(z)       per-dim normalise → τ map
#   3. τ_k = √ᾱ_k·τ_0 + √(1−ᾱ_k)·ε                 forward diffusion
#   4. denoiser predicts v from τ_k
#   5. τ̂_0 = √ᾱ_k·τ_k − √(1−ᾱ_k)·v̂_k              inverse v-prediction
#   6. z̃   = LatentTauTransform.from_tau_space(τ̂_0)   invert τ → z
#   7. K̂*  = vae.decode(z̃, physics, target_len)

# Noise schedule: cosine ᾱ (Nichol & Dhariwal 2021).
# As k→T: ᾱ_k → alpha_bar_min ≈ 0, so τ_k → ε (pure Gaussian noise).
# The τ reparameterisation is an inductive bias on latent geometry — clear days
# encode near τ≈0, overcast near large τ.

# Affine standardisation:
#   Raw τ ∈ [0, τ_max] has std ≈ 0.48, which compresses signal relative to the
#   unit-Gaussian noise prior.  LatentTauTransform.fit() measures τ_mu and τ_sig
#   from the training set and standardises so the diffused distribution has
#   mean≈0 and std≈1.

#   The clamp bounds used in p_sample / p_sample_ddim are kept SYMMETRIC around
#   zero in standardised space:
#       tau_min_std = −tau_max_std   where  tau_max_std = (τ_max_raw − τ_mu) / τ_sig
#   This prevents asymmetric clamping from biasing the reverse chain toward the
#   overcast end of τ-space (the core bug fixed here).  Values that fall below
#   −tau_max_std decode to physical τ < 0 and are harmlessly clamped to zero by
#   from_tau_space(), so the Beer-Lambert constraint is still enforced.

#   Always call latent_transform.tau_clamp_bounds() and pass its result as
#   (tau_min, tau_max) to p_sample / p_sample_ddim.  Never use the raw defaults.

# v-prediction (Salimans & Ho 2022):
#   v_k   = √ᾱ_k·ε − √(1−ᾱ_k)·τ_0
#   τ̂_0  = √ᾱ_k·τ_k − √(1−ᾱ_k)·v̂_k


import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


#
# Primal maps: K* ↔ τ
#

def kstar_to_tau(k: torch.Tensor, k_max: float, eps: float = 1e-6) -> torch.Tensor:
    """K* → τ = −log(K* / k_max).  Returns τ ≥ 0."""
    k_safe = k.clamp(eps, k_max)
    return -torch.log(k_safe / k_max)


def tau_to_kstar(tau: torch.Tensor, k_max: float) -> torch.Tensor:
    """τ → K* = k_max · exp(−τ).  Returns K* ∈ (0, k_max]."""
    return (k_max * torch.exp(-tau.clamp(min=0.0))).clamp(0.0, k_max)


# 
# LatentTauTransform — per-dimension normalisation + τ map for z-space
# 

class LatentTauTransform:
    # Per-dimension normalisation of AE latents followed by τ mapping.

    # Normalises each latent dimension j to (0, k_max] using per-dimension
    # bounds fitted from the training set, then applies τ = −log(z_norm/k_max).

    # Polarity correction: the encoder assigns dimension directions arbitrarily.
    # Dims where clear days have lower mean z than cloudy days produce inverted
    # tau (clear > cloudy tau).  fit() detects these dims from regime_labels and
    # stores a flip mask; to_tau_space() reflects those dims before normalisation
    # so that high-z = clear universally.  from_tau_space() undoes the flip.

    # Affine standardisation: fit() computes τ_mu (mean over training samples)
    # and τ_sig (std) in raw τ-space and standardises so the diffusion prior
    # N(0,1) matches the data distribution.  Both are scalar — per-dimension
    # stats would require storing a (d_z,) vector and re-fitting the denoiser;
    # a scalar correction is sufficient to fix the 2× noise-signal compression
    # without altering the model architecture.

    # CLAMP SYMMETRY CONTRACT
    # -----------------------
    # tau_clamp_bounds() returns (tau_min_std, tau_max_std) where
    #     tau_min_std = −tau_max_std   (symmetric around 0)
    # This is intentional.  The asymmetric alternative
    #     tau_min_std = (0 − tau_mu) / tau_sig   (physical lower bound)
    # clips the clear-sky tail only ~1.5 standardised units below zero while
    # allowing ~4.25 units of headroom on the overcast side, which biases the
    # reverse chain toward mixed/overcast.  The symmetric clamp lets values
    # overshoot the physical floor in standardised space; from_tau_space() then
    # applies tau.clamp(min=0) to enforce Beer-Lambert before the exp-map, so
    # the physical constraint is preserved at the correct stage.

    # τ range after normalisation (raw): [0, τ_max] where τ_max = −log(z_norm_clamp_lo).
    # τ range after standardisation: centred near 0, std ≈ 1.

    # Workflow:
    #     transform = LatentTauTransform(k_max=cfg["physics"]["k_max"])
    #     transform.fit(latents, regime_labels)
    #     transform.save(cfg["paths"]["latent_tau_stats"])
    #     # --- Stage 2 / generation ---
    #     transform.load(cfg["paths"]["latent_tau_stats"])
    #     tau0           = transform.to_tau_space(z)
    #     tau_min, tau_max = transform.tau_clamp_bounds()
    #     z_hat          = transform.from_tau_space(tau_hat)
    # 

    def __init__(self, k_max: float, z_norm_clamp_lo: float = 0.01):
        self.k_max           = k_max
        self.z_norm_clamp_lo = z_norm_clamp_lo   # lower clamp on z_norm; sets τ_max = −log(z_norm_clamp_lo)
        self._z_min     = None   # (d_z,) lower bound per dimension
        self._z_max     = None   # (d_z,) upper bound per dimension
        self._flip_dims = None   # (d_z,) bool — dims where clear < cloudy mean
        # Affine standardisation scalars.  Identity defaults (mu=0, sig=1) so
        # old checkpoints saved without these fields remain forward-compatible.
        self._tau_mu    = 0.0    # mean of raw τ over training set
        self._tau_sig   = 1.0    # std  of raw τ over training set  (≥ 0.01)
        # Per-split affine stats for z_flat only (dims 0:d_z_flat).
        # Stored separately so from_tau_space() can invert only z_flat without
        # the z_var dims inflating tau_mu / deflating tau_sig.
        # None = not stored (old checkpoint) → fall back to global stats.
        self._tau_mu_flat  = None   # float or None
        self._tau_sig_flat = None   # float or None
        self._d_z_flat     = None   # int — how many dims are z_flat; None = unknown
        self._fitted    = False

    def fit(
        self,
        z_array: np.ndarray,
        regime_labels: Optional[np.ndarray] = None,
        percentile_lo: float = 1.0,
        percentile_hi: float = 99.0,
        std_margin_factor: float = 0.5,
        polarity_min_samples: int = 10,
        d_z_flat: Optional[int] = None,   # cfg["vae"]["latent_dim"] — enables per-split affine stats
    ) -> None:
        # Fit per-dimension bounds from a (N, d_z) array of training latents.

        # d_z_flat : number of z_flat dimensions (cfg["vae"]["latent_dim"]).
        #     When provided, separate affine standardisation stats are computed
        #     for z_flat dims (0:d_z_flat) vs the full array. from_tau_space()
        #     uses these when inverting only z_flat to avoid z_var's narrower
        #     range corrupting the tau_mu / tau_sig estimates.
        #     Pass cfg["vae"]["latent_dim"] here from train_diffusion().

        # Uses percentile-based bounds (p1/p99 default) plus a margin of
        # std_margin_factor × per-dim std, making the transform robust to
        # outliers and latent mean drift.

        # Polarity correction: if regime_labels (0=clear, 1+=cloudy) is provided,
        # dims where mean_z[clear] < mean_z[cloudy] are stored in _flip_dims and
        # reflected in to_tau_space so that high-z = clear universally.

        # Affine standardisation: τ_mu and τ_sig are computed as the mean and std
        # of raw τ values over the entire training set (after polarity flip and
        # log-map, before standardisation).  The mean is computed per-sample first
        # and then averaged, so that latent dimensions with systematically higher
        # tau do not inflate the scalar unduly.  τ_sig is clamped to ≥ 0.01.
        # 
        if z_array.ndim != 2:
            raise ValueError(f"z_array must be 2-D (N, d_z), got {z_array.shape}")
        d_z = z_array.shape[1]

        lo     = np.percentile(z_array, percentile_lo, axis=0).astype(np.float32)
        hi     = np.percentile(z_array, percentile_hi, axis=0).astype(np.float32)
        std    = z_array.std(axis=0).astype(np.float32)
        margin = std_margin_factor * std

        self._z_min = lo - margin
        self._z_max = hi + margin
        min_range   = np.maximum(std * 0.1, 1e-3)
        too_narrow  = (self._z_max - self._z_min) < min_range
        self._z_max[too_narrow] = self._z_min[too_narrow] + min_range[too_narrow]

        if regime_labels is not None and len(regime_labels) == len(z_array):
            is_clear  = (regime_labels == 0)
            is_cloudy = (regime_labels >= 1)
            if is_clear.sum() >= polarity_min_samples and is_cloudy.sum() >= polarity_min_samples:
                mean_clear  = z_array[is_clear].mean(axis=0)
                mean_cloudy = z_array[is_cloudy].mean(axis=0)
                self._flip_dims = (mean_clear < mean_cloudy).astype(np.bool_)
                log.info(
                    "  Polarity correction: %d/%d dims flipped",
                    int(self._flip_dims.sum()), d_z,
                )
            else:
                self._flip_dims = np.zeros(d_z, dtype=np.bool_)
        else:
            self._flip_dims = np.zeros(d_z, dtype=np.bool_)
            if regime_labels is None:
                log.warning(
                    "  Polarity correction skipped: no regime_labels provided."
                )

        # Compute affine standardisation parameters.
        # Replicate the exact polarity-flip + log-map pipeline of to_tau_space()
        # so that _tau_mu / _tau_sig describe the actual standardised output.
        # Per-sample mean is computed first (axis=1), then averaged over N,
        # so dimensions with systematically high τ do not inflate the scalar.
        _z_for_stats = z_array.astype(np.float32).copy()
        if self._flip_dims.any():
            _denom_f = np.maximum(self._z_max - self._z_min, 1e-8)
            _mid_f   = self._z_min + 0.5 * _denom_f
            _z_for_stats[:, self._flip_dims] = (
                2.0 * _mid_f[self._flip_dims] - _z_for_stats[:, self._flip_dims]
            )
        _denom_np  = np.maximum(self._z_max - self._z_min, 1e-8)
        _z_norm_np = (_z_for_stats - self._z_min) / _denom_np
        _z_norm_np = np.clip(_z_norm_np, self.z_norm_clamp_lo, 1.0) * self.k_max
        _tau_np    = -np.log(_z_norm_np / self.k_max)      # (N, d_z) raw τ values

        # Use per-sample mean (mean over d_z) then aggregate over N.
        # This prevents high-τ dims from dominating the scalar estimator.
        _sample_means = _tau_np.mean(axis=1)               # (N,) one scalar per sample
        self._tau_mu  = float(_sample_means.mean())
        self._tau_sig = float(max(_tau_np.std(), 0.01))    # global std; clamped ≥ 0.01

        # Per-split affine stats: compute separately for z_flat dims only.
        # z_var dims (appended after z_flat) have a ~4× narrower value range,
        # so they inflate _tau_mu and deflate _tau_sig for the z_flat dims.
        # Storing separate stats means from_tau_space() uses the correct
        # normalisation when inverting only z_flat (first d_z_flat dims).
        # d_z_flat is passed via the optional keyword; if None we store None
        # and fall back to global stats (backward compatible).
        if d_z_flat is not None and 0 < d_z_flat < d_z:
            self._d_z_flat    = int(d_z_flat)
            _tau_flat         = _tau_np[:, :d_z_flat]      # (N, d_z_flat)
            _flat_means       = _tau_flat.mean(axis=1)     # (N,)
            self._tau_mu_flat  = float(_flat_means.mean())
            self._tau_sig_flat = float(max(_tau_flat.std(), 0.01))
            log.info(
                "  Per-split affine stats (z_flat dims 0:%d): "
                "tau_mu_flat=%.4f  tau_sig_flat=%.4f  "
                "(global: tau_mu=%.4f  tau_sig=%.4f)",
                d_z_flat,
                self._tau_mu_flat, self._tau_sig_flat,
                self._tau_mu, self._tau_sig,
            )
        else:
            self._d_z_flat     = None
            self._tau_mu_flat  = None
            self._tau_sig_flat = None
            if d_z_flat is not None:
                log.warning(
                    "fit(): d_z_flat=%s is out of range for d_z=%d — "
                    "per-split affine stats not stored. Pass cfg['vae']['latent_dim'] "
                    "as d_z_flat to enable z_flat-specific tau normalisation.",
                    d_z_flat, d_z,
                )

        self._fitted = True
        log.info(
            "LatentTauTransform fitted | d_z=%d | flip=%d/%d | "
            "z_min [%.4f, %.4f] | z_max [%.4f, %.4f] | "
            "tau_mu=%.4f  tau_sig=%.4f  tau_max_raw=%.4f",
            d_z, int(self._flip_dims.sum()), d_z,
            float(self._z_min.min()), float(self._z_min.max()),
            float(self._z_max.min()), float(self._z_max.max()),
            self._tau_mu, self._tau_sig,
            float(-np.log(self.z_norm_clamp_lo)),
        )

    def save(self, path: str | Path) -> None:
        """Persist fitted statistics to JSON."""
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k_max":           float(self.k_max),
            "z_norm_clamp_lo": float(self.z_norm_clamp_lo),
            "z_min":           self._z_min.tolist(),
            "z_max":           self._z_max.tolist(),
            "flip_dims":       self._flip_dims.tolist(),
            "tau_mu":          float(self._tau_mu),
            "tau_sig":         float(self._tau_sig),
            # Per-split stats for z_flat dims — None if not fitted with d_z_flat.
            # from_tau_space() uses these when inverting only z_flat dims to avoid
            # z_var's narrower range corrupting tau normalisation.
            "d_z_flat":        self._d_z_flat,
            "tau_mu_flat":     self._tau_mu_flat,
            "tau_sig_flat":    self._tau_sig_flat,
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("LatentTauTransform saved → %s", path)

    def load(self, path: str | Path) -> None:
        #Load statistics from a previously saved JSON file
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"LatentTauTransform stats not found: {path.resolve()}.  "
                "Run fit_latent_tau_transform() after Stage 1 training."
            )
        with path.open() as f:
            blob = json.load(f)
        self.k_max           = float(blob["k_max"])
        self.z_norm_clamp_lo = float(blob.get("z_norm_clamp_lo", 0.01))
        self._z_min     = np.array(blob["z_min"],  dtype=np.float32)
        self._z_max     = np.array(blob["z_max"],  dtype=np.float32)
        self._flip_dims = np.array(
            blob.get("flip_dims", [False] * len(self._z_min)), dtype=np.bool_
        )
        # Identity defaults (0, 1) keep old checkpoints forward-compatible.
        self._tau_mu  = float(blob.get("tau_mu",  0.0))
        self._tau_sig = float(blob.get("tau_sig", 1.0))
        # Per-split stats — None if saved by an older version without d_z_flat.
        self._d_z_flat     = blob.get("d_z_flat",     None)
        self._tau_mu_flat  = blob.get("tau_mu_flat",  None)
        self._tau_sig_flat = blob.get("tau_sig_flat", None)
        if self._d_z_flat is not None:
            self._d_z_flat = int(self._d_z_flat)
        self._fitted  = True
        log.info(
            "LatentTauTransform loaded ← %s  (tau_mu=%.4f  tau_sig=%.4f  "
            "tau_max_raw=%.4f  flip_dims=%d/%d  d_z_flat=%s  "
            "tau_mu_flat=%s  tau_sig_flat=%s)",
            path, self._tau_mu, self._tau_sig,
            float(-np.log(self.z_norm_clamp_lo)),
            int(self._flip_dims.sum()), len(self._flip_dims),
            self._d_z_flat,
            f"{self._tau_mu_flat:.4f}"  if self._tau_mu_flat  is not None else "None",
            f"{self._tau_sig_flat:.4f}" if self._tau_sig_flat is not None else "None",
        )

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "LatentTauTransform has not been fitted.  "
                "Call fit() or load() before to_tau_space / from_tau_space."
            )

    def validate_against(
        self,
        z_sample: np.ndarray,
        tau_mean_tol: float = 0.5,
        tau_std_tol: float  = 0.4,
    ) -> bool:
        # FIX C6: Check whether loaded affine stats match the current encoder output.

        # The affine stats (tau_mu, tau_sig) are fitted once on the training latent
        # cache and frozen. If z_std drifted during VAE training, the saved stats are
        # wrong for the current encoder — causing 2-3× tau_std miscalibration that
        # degrades the SNR at every diffusion step.

        # Args:
        #     z_sample    : (N, d_z) float32 array of current-encoder latent codes.
        #     tau_mean_tol: warn if |observed_tau_mean - fitted_tau_mu| > this.
        #     tau_std_tol : warn if |observed_tau_std/fitted_tau_sig - 1| > this.

        # Returns True if stats match within tolerance, False if re-fitting is needed.
        # Called by train_diffusion() after loading the transform on resume.
        
        self._check()
        import torch as _torch
        z_t   = _torch.from_numpy(z_sample.astype(np.float32))
        tau_t = self.to_tau_space(z_t)  # uses existing affine stats
        obs_mean = float(tau_t.mean().item())
        obs_std  = float(tau_t.std().item())

        mean_err = abs(obs_mean)               # should be ~0 after standardisation
        std_err  = abs(obs_std - 1.0)          # should be ~1 after standardisation

        ok = True
        if mean_err > tau_mean_tol:
            log.warning(
                "[C6] LatentTauTransform.validate_against: "
                "tau_mean=%.3f is far from 0 (tolerance=%.2f). "
                "The affine stats (tau_mu=%.4f) were fitted on a different encoder "
                "distribution. Re-fit by deleting the diffusion checkpoint and re-running "
                "train_diffusion, or rebuild the latent cache manually.",
                obs_mean, tau_mean_tol, self._tau_mu,
            )
            ok = False
        if std_err > tau_std_tol:
            log.warning(
                "[C6] LatentTauTransform.validate_against: "
                "tau_std=%.3f is far from 1.0 (tolerance=%.2f, err=%.2f×). "
                "The affine scale (tau_sig=%.4f) is stale — z_std likely drifted "
                "during VAE training (e.g. collapse from %.2f to %.2f). "
                "This causes %.1f× SNR miscalibration at mid-noise steps. "
                "ACTION: rebuild latent cache and re-fit tau transform.",
                obs_std, tau_std_tol, std_err,
                self._tau_sig,
                self._tau_sig, self._tau_sig * obs_std,  # approx original vs current z_std
                max(obs_std, 1.0 / max(obs_std, 1e-4)),  # amplification factor
            )
            ok = False
        if ok:
            log.info(
                "[C6] LatentTauTransform.validate_against: stats match current encoder "
                "(tau_mean=%.3f, tau_std=%.3f). No re-fitting needed.",
                obs_mean, obs_std,
            )
        return ok

    def tau_clamp_bounds(self) -> Tuple[float, float]:
        # Return (tau_min_std, tau_max_std) for clamping τ̂₀ in p_sample / p_sample_ddim.

        # The bounds are SYMMETRIC around zero in standardised space:
        #     tau_max_std = (τ_max_raw − τ_mu) / τ_sig
        #     tau_min_std = −tau_max_std

        # Symmetry is critical.  Using the physical lower bound as tau_min_std:
        #     tau_min_std = (0 − τ_mu) / τ_sig   ← DO NOT USE
        # produces an asymmetric clamp (e.g. [−1.5, +4.25]) that clips the
        # clear-sky tail 3× more aggressively than the overcast tail, biasing
        # the reverse chain toward mixed/overcast profiles.

        # The symmetric clamp allows τ̂₀ to go slightly below −tau_max_std in
        # standardised space (i.e. physically below τ=0).  from_tau_space()
        # applies tau.clamp(min=0) before the exp-map, so Beer-Lambert is still
        # satisfied — the physical constraint is enforced at the right stage.

        # Always call this method and pass its output explicitly to p_sample /
        # p_sample_ddim.  Never rely on the default argument values of those
        # methods, which are stale placeholders.
        # 
        self._check()
        tau_max_raw = float(-np.log(self.z_norm_clamp_lo))   # e.g. −log(0.01) = 4.605
        tau_max_std = (tau_max_raw - self._tau_mu) / self._tau_sig
        tau_min_std = -tau_max_std   # symmetric — prevents overcast bias
        return tau_min_std, tau_max_std

    def to_tau_space(self, z: torch.Tensor) -> torch.Tensor:
        # Normalise z → (0, k_max] per-dimension (with polarity flip), then
        # apply τ = −log(·/k_max) and affine standardisation.

        # Returns standardised τ with mean≈0, std≈1 over the training set.
        # z : (..., d_z) AE latent codes, any leading batch shape.

        # Emits a warning when > 5% of dimensions hit the z_norm_clamp_lo lower
        # bound — this signals latent codes outside the training-set distribution
        # (unfitted transform or distribution shift) and those dims are forced to
        # the τ_max extreme (heavy-overcast end).
        # 
        self._check()
        dev   = z.device
        z_min = torch.from_numpy(self._z_min).to(dev)
        z_max = torch.from_numpy(self._z_max).to(dev)
        flip  = torch.from_numpy(self._flip_dims).to(dev)
        denom = (z_max - z_min).clamp(min=1e-8)

        z = z.clone()
        # Reflect flipped dims around their per-dimension midpoint so that
        # high-z = clear universally after the flip.
        mid = z_min + 0.5 * denom
        z[..., flip] = 2.0 * mid[flip] - z[..., flip]

        z_norm = (z - z_min) / denom

        # Diagnostic: fraction of entries below the lower clamp.
        clamp_lo   = self.z_norm_clamp_lo
        clamp_hits = (z_norm < clamp_lo).float().mean().item()
        if clamp_hits > 0.05:
            log.warning(
                "LatentTauTransform.to_tau_space: %.1f%% of latent entries are "
                "below z_norm_clamp_lo=%.3f and will be clamped to τ_max=%.2f. "
                "Re-fit LatentTauTransform on the current AE checkpoint.",
                100.0 * clamp_hits, clamp_lo, -np.log(clamp_lo),
            )

        z_norm = z_norm.clamp(clamp_lo, 1.0) * self.k_max
        tau    = -torch.log(z_norm / self.k_max)   # raw τ ∈ [0, τ_max]
        # Affine standardisation: shift and scale to mean≈0, std≈1.
        return (tau - self._tau_mu) / self._tau_sig

    def from_tau_space(self, tau: torch.Tensor) -> torch.Tensor:
        # Invert standardised τ → z (original AE scale), undoing polarity flip.

        # tau : (..., d_in) standardised optical-depth codes from the denoiser.
        #       d_in may be less than the total fitted dimensionality — for
        #       example when only z_flat dims ([:d_z_flat]) are passed, excluding
        #       z_var dims, as part of the z_var saturation fix.  The fitted
        #       arrays _z_min / _z_max / _flip_dims are sliced to [:d_in]
        #       automatically so the caller does not need to know the exact split.

        # Affine stats selection:
        #   If d_in == _d_z_flat and per-split stats (_tau_mu_flat, _tau_sig_flat)
        #   are available, those are used instead of the global stats.  This is
        #   critical because z_var dims (appended after z_flat) have a ~4× narrower
        #   value range; their tau values inflate _tau_mu and deflate _tau_sig,
        #   causing the global affine inversion to produce wrong z_flat values.
        #   Per-split stats avoid this without requiring architectural changes.

        # Inverse pipeline:
        #   1. Undo standardisation: τ_raw = τ_std × τ_sig + τ_mu
        #   2. Clamp τ_raw ≥ 0  (Beer-Lambert requires τ ≥ 0; v-prediction can
        #      produce slightly negative values — clamped here, not in p_sample)
        #   3. Invert log-map: z_norm = k_max · exp(−τ_raw)
        #   4. Invert per-dim normalisation: z = z_norm/k_max × denom + z_min
        #   5. Undo polarity flip on affected dims
        #   6. Safety clamp z to [z_min, z_max]
        # 
        self._check()
        dev  = tau.device
        d_in  = tau.shape[-1]
        z_min = torch.from_numpy(self._z_min[:d_in]).to(dev)
        z_max = torch.from_numpy(self._z_max[:d_in]).to(dev)
        flip  = torch.from_numpy(self._flip_dims[:d_in]).to(dev)
        denom = (z_max - z_min).clamp(min=1e-8)

        # Select affine stats: use per-split (z_flat-only) when available and
        # the input dimensionality matches d_z_flat exactly.
        if (
            self._d_z_flat is not None
            and self._tau_mu_flat  is not None
            and self._tau_sig_flat is not None
            and d_in == self._d_z_flat
        ):
            tau_mu  = self._tau_mu_flat
            tau_sig = self._tau_sig_flat
            log.debug(
                "from_tau_space: using z_flat-specific affine stats "
                "(tau_mu_flat=%.4f  tau_sig_flat=%.4f  d_in=%d)",
                tau_mu, tau_sig, d_in,
            )
        else:
            tau_mu  = self._tau_mu
            tau_sig = self._tau_sig
            if self._d_z_flat is not None and d_in != self._d_z_flat:
                log.debug(
                    "from_tau_space: d_in=%d != d_z_flat=%d — using global "
                    "affine stats (tau_mu=%.4f  tau_sig=%.4f).",
                    d_in, self._d_z_flat, tau_mu, tau_sig,
                )

        # Step 1–3: undo standardisation then invert log-map.
        tau_raw = tau * tau_sig + tau_mu
        # Step 2: clamp physical τ ≥ 0.  Values that overshoot the physical
        # floor (τ < 0) map to K* > k_max; clamping here is the correct place
        # because p_sample uses a symmetric clamp that intentionally allows
        # standardised τ to go below −tau_max_std during the reverse chain.
        z_norm = (self.k_max * torch.exp(-tau_raw.clamp(min=0.0))).clamp(0.0, self.k_max)

        # Step 4: invert per-dim normalisation.
        z = (z_norm / self.k_max) * denom + z_min

        # Step 5: undo polarity flip.
        z = z.clone()
        mid = z_min + 0.5 * denom
        z[..., flip] = 2.0 * mid[flip] - z[..., flip]

        # Step 6: safety clamp — guards floating-point edge cases after flip.
        z = torch.clamp(z, min=z_min, max=z_max)
        return z



# Cosine α̅ noise schedule

class TauNoiseSchedule:
    # Cosine (or linear) α̅ noise schedule for diffusion in τ-space.

    # Forward process: τ_k = √ᾱ_k·τ_0 + √(1−ᾱ_k)·ε
    # As k→T: ᾱ_k → alpha_bar_min ≈ 0, so τ_k → ε (pure Gaussian noise).
    

    def __init__(
        self,
        num_steps: int,
        alpha_bar_min: float,
        schedule: str = "cosine",
        device: torch.device = torch.device("cpu"),
        cosine_s: float = 0.008,  # offset for cosine schedule (Nichol & Dhariwal 2021)
    ):
        self.T      = num_steps
        self.device = device
        alpha_bars  = self._build_schedule(num_steps, alpha_bar_min, schedule, cosine_s=cosine_s)
        self._register(alpha_bars, device)

    def _build_schedule(
        self, T: int, alpha_bar_min: float, schedule: str, cosine_s: float = 0.008
    ) -> torch.Tensor:
        steps = torch.arange(T + 1, dtype=torch.float64)
        if schedule == "cosine":
            s = cosine_s
            f = torch.cos(((steps / T + s) / (1.0 + s)) * (torch.pi / 2.0)) ** 2
            alpha_bars = f / f[0]
            lo, hi = float(alpha_bars[-1]), float(alpha_bars[0])
            alpha_bars = alpha_bar_min + (alpha_bars - lo) * (
                (1.0 - alpha_bar_min) / (hi - lo + 1e-12)
            )
        elif schedule == "linear":
            alpha_bars = torch.linspace(1.0, alpha_bar_min, T + 1, dtype=torch.float64)
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Use 'cosine' or 'linear'."
            )
        return alpha_bars.float()[1:]   # (T,) — index 0 corresponds to step 1

    def _register(self, alpha_bars: torch.Tensor, device: torch.device) -> None:
        self.alpha_bars      = alpha_bars.to(device)
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus  = (1.0 - self.alpha_bars).sqrt()

    def to(self, device: torch.device) -> "TauNoiseSchedule":
        self.device = device
        self._register(self.alpha_bars.cpu(), device)
        return self

    def q_sample(
        self,
        tau0: torch.Tensor,
        k: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward process: τ_k = √ᾱ_k·τ_0 + √(1−ᾱ_k)·ε.  Returns (tau_k, noise).

        # k may be (B,) — one step per window — or (B, W) — one step per day.
        # tau0 is (B, W, d_z). Coefficients broadcast over the d_z dimension.
        
        if noise is None:
            noise = torch.randn_like(tau0)
        k = k.long()
        if k.dim() == 1:
            # (B,) → one noise level shared across all W days in each window
            sab = self.sqrt_alpha_bars[k].view(-1, 1, 1)    # (B, 1, 1)
            som = self.sqrt_one_minus[k].view(-1, 1, 1)
        else:
            # (B, W) → independent noise level per day
            flat = k.reshape(-1)                             # (B*W,)
            sab  = self.sqrt_alpha_bars[flat].view(k.shape[0], k.shape[1], 1)
            som  = self.sqrt_one_minus[flat].view(k.shape[0], k.shape[1], 1)
        return sab * tau0 + som * noise, noise

    def get_v_target(
        self,
        tau0: torch.Tensor,
        noise: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        # v-prediction target: v = √ᾱ_k·ε − √(1−ᾱ_k)·τ_0.

        # k may be (B,) or (B, W) — same broadcasting logic as q_sample.
        
        k = k.long()
        if k.dim() == 1:
            sab = self.sqrt_alpha_bars[k].view(-1, 1, 1)
            som = self.sqrt_one_minus[k].view(-1, 1, 1)
        else:
            flat = k.reshape(-1)
            sab  = self.sqrt_alpha_bars[flat].view(k.shape[0], k.shape[1], 1)
            som  = self.sqrt_one_minus[flat].view(k.shape[0], k.shape[1], 1)
        return sab * noise - som * tau0

    @torch.no_grad()
    def p_sample(
        self,
        tau_k:   torch.Tensor,   # (B, W, d_z) noisy τ at step k
        k:       int,             # current step index (0-indexed)
        v_pred:  torch.Tensor,   # (B, W, d_z) predicted v
        tau_max: float = 4.605,  # upper clamp in standardised τ-space
        tau_min: float = -4.605, # lower clamp in standardised τ-space (symmetric)
    ) -> torch.Tensor:           # (B, W, d_z) denoised τ at step k-1
        # Single DDPM reverse step: τ_k → τ_{k-1}.

        # Recovers τ̂_0 from v-prediction, clamps it to the valid τ domain, then
        # samples the posterior q(τ_{k-1} | τ_k, τ̂_0).  Returns τ̂_0 at k=0.

        # IMPORTANT — always pass tau_min and tau_max from
        # latent_transform.tau_clamp_bounds(), not the default values.  The
        # defaults are symmetric placeholders matching z_norm_clamp_lo=0.01 with
        # identity standardisation (tau_mu=0, tau_sig=1); they will be wrong for
        # any fitted transform with non-identity parameters.

        # The clamp is intentionally SYMMETRIC: tau_min = −tau_max.  This
        # prevents the reverse chain from being biased toward the overcast end
        # of τ-space.  See LatentTauTransform.tau_clamp_bounds() for details.

        # Schedule is 0-indexed: alpha_bars[i] = ā_{i+1}.
        # At k=1, ā_0 = 1.0 (no variance at t=0).
        
        ab_k  = self.alpha_bars[k]
        sab_k = self.sqrt_alpha_bars[k]
        som_k = self.sqrt_one_minus[k]

        tau0_hat = sab_k * tau_k - som_k * v_pred
        # Symmetric clamp in standardised τ-space.
        # Physical enforcement (τ ≥ 0) is handled inside from_tau_space().
        tau0_hat = tau0_hat.clamp(tau_min, tau_max)

        if k == 0:
            return tau0_hat

        ab_prev    = torch.tensor(1.0, device=tau_k.device) if k == 1 else self.alpha_bars[k - 1]
        coef1      = ab_prev.sqrt() * (1.0 - ab_k / ab_prev) / (1.0 - ab_k + 1e-8)
        coef2      = (ab_k / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab_k + 1e-8)
        mu         = coef1 * tau0_hat + coef2 * tau_k
        beta_tilde = (1.0 - ab_prev) / (1.0 - ab_k + 1e-8) * (1.0 - ab_k / ab_prev)
        return mu + beta_tilde.sqrt() * torch.randn_like(tau_k)

    def build_ddim_steps(self, ddim_steps: int) -> List[int]:
        # Return a list of `ddim_steps` evenly-spaced timestep indices for DDIM.

        # Indices are in descending order (T-1 down to 0), i.e. the order they
        # are visited during the reverse process.  The final step is always 0
        # so that τ̂_0 is returned exactly without a posterior step.

        # Example: T=1000, ddim_steps=50 → [999, 979, 959, ..., 19, 0]
        
        if ddim_steps >= self.T:
            return list(reversed(range(self.T)))
        indices = [round(i * (self.T - 1) / (ddim_steps - 1))
                   for i in range(ddim_steps)]
        indices = sorted(set(indices), reverse=True)
        if indices[-1] != 0:
            indices.append(0)
        return indices

    @torch.no_grad()
    def p_sample_ddim(
        self,
        tau_k:   torch.Tensor,   # (B, W, d_z) noisy τ at step k
        k:       int,             # current step index (0-indexed)
        k_prev:  int,             # previous step index (< k; −1 means final step)
        v_pred:  torch.Tensor,   # (B, W, d_z) predicted v
        eta:     float = 0.0,    # stochasticity: 0.0 = fully deterministic DDIM
        tau_max: float = 4.605,  # upper clamp in standardised τ-space
        tau_min: float = -4.605, # lower clamp in standardised τ-space (symmetric)
    ) -> torch.Tensor:           # (B, W, d_z) denoised τ at step k_prev
        # Single DDIM reverse step: τ_k → τ_{k_prev}  (Song et al. 2021).

        # With eta=0 (default) this is fully deterministic.  With eta=1 it
        # recovers DDPM-like stochasticity.

        # IMPORTANT — always pass tau_min and tau_max from
        # latent_transform.tau_clamp_bounds().  The defaults are symmetric
        # placeholders for identity standardisation only.

        # The clamp on τ̂₀ is SYMMETRIC (tau_min = −tau_max) to prevent
        # overcast bias.  from_tau_space() enforces τ_raw ≥ 0 via clamp(min=0)
        # before the exp-map, so Beer-Lambert is still satisfied.

        # ε̂ is re-derived from the clamped τ̂₀ so that the DDIM direction
        # vector is self-consistent with the clamped prediction:
        #     ε̂ = (τ_k − √ᾱ_k · τ̂₀) / √(1−ᾱ_k)

        # At the final step (k_prev == −1) returns τ̂_0 directly.

        # Reference: Song et al. (2021) "Denoising Diffusion Implicit Models"
        # https://arxiv.org/abs/2010.02502

        # v-prediction form (Salimans & Ho 2022):
        #     τ̂_0  = √ᾱ_k · τ_k − √(1−ᾱ_k) · v̂
        #     ε̂   = √ᾱ_k · v̂  + √(1−ᾱ_k) · τ_k
        #     τ_{k_prev} = √ᾱ_{k_prev} · τ̂_0
        #                  + √(1−ᾱ_{k_prev} − σ²) · ε̂
        #                  + σ · ε   (ε ~ N(0,I) if eta > 0 else 0)
        
        ab_k  = self.alpha_bars[k]
        sab_k = self.sqrt_alpha_bars[k]
        som_k = self.sqrt_one_minus[k]

        # Recover τ̂_0 from v-prediction and apply symmetric clamp.
        tau0_hat = sab_k * tau_k - som_k * v_pred
        tau0_hat = tau0_hat.clamp(tau_min, tau_max)

        # Re-derive ε̂ from the clamped τ̂_0 so that the DDIM direction remains
        # self-consistent with tau0_hat (prevents noise direction drift in the
        # clamped region).
        eps_hat = (tau_k - sab_k * tau0_hat) / (som_k + 1e-8)

        if k_prev < 0:
            # Final step: return clean τ̂_0 directly (no posterior step).
            return tau0_hat

        ab_prev = self.alpha_bars[k_prev]

        # DDIM posterior variance (eta=0 → deterministic; eta=1 → DDPM-like)
        sigma = (eta
                 * ((1.0 - ab_prev) / (1.0 - ab_k + 1e-8)).sqrt()
                 * (1.0 - ab_k / (ab_prev + 1e-8)).sqrt())

        dir_coef = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt()
        mu       = ab_prev.sqrt() * tau0_hat + dir_coef * eps_hat

        if eta > 0.0:
            return mu + sigma * torch.randn_like(tau_k)
        return mu