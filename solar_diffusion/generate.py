# 
# solar_diffusion/generate.py
# ----------------------------
# Arbitrary-length K* sequence generation.

# Pipeline
# --------
# 1. Compute deterministic solar geometry for every requested day (pvlib).
# 2. If N <= W: single DDPM reverse pass.
# 3. If N > W:  autoregressive sliding window with (W - S) days of context overlap.
# 4. Convert generated tau latents -> z using LatentTauTransform.from_tau_space().
# 5. Decode each latent z -> sunlit K* profile via the AE decoder.
# 6. Post-process: clip to [0, k_max], insert night zeros, smooth sunrise/sunset ramps.

# Regime conditioning — self-consistent feedback loop
# ----------------------------------------------------
# The RegimeEmbedding inside SolarDenoiser was trained with real GMM-derived regime
# labels.  At generation time we have no ground-truth labels, so we infer them
# directly from the model's own predictions using a feedback loop:

#   At each denoising step k:
#     1. Run a cheap forward pass to predict τ̂₀ from the current noisy τₖ.
#     2. Map τ̂₀ → per-day τ means (mean over d_z).
#     3. Convert τ means to soft regime probabilities via the GMM τ-space boundaries
#        (clear ↔ low τ, overcast ↔ high τ) fitted from the training latent cache.
#     4. Take argmax → hard regime labels (1, W) int64.
#     5. Feed those labels back as `regime_ids` to the full CFG forward pass.

# This is self-consistent: the regime signal the denoiser receives at each step
# reflects where the trajectory is actually heading, not a hardcoded prior.
# Generalisation comes from the physics conditioning (location, DOY, ETR) — the
# denoiser learned that a desert site in summer naturally stays clear, a coastal
# site in winter stays mixed/overcast.  The regime labels track what the model is
# generating and reinforce it, rather than imposing it.

# The GMM τ-space boundaries are fitted once from the training latent cache during
# Stage 2 (stored in latent_tau_stats.json).  If they are unavailable the loop
# falls back to null regime tokens (original behaviour, no conditioning).

# No meteorological observations are used at inference.
# User input: start_date, end_date, lat, lon.
# Everything else is computed from pvlib and the trained model weights.
# 

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from solar_diffusion.denoiser import SolarDenoiser
from solar_diffusion.optical_depth import LatentTauTransform, TauNoiseSchedule
from solar_diffusion.vae import SolarVAE

log = logging.getLogger(__name__)


# 
# Self-consistent 4-class regime inference from τ̂₀ predictions
# 

def _load_tau_class_centroids(
    latent_transform: "LatentTauTransform",
    cfg: Dict,
) -> Optional[np.ndarray]:
    # Load per-class τ centroids written by fit_latent_tau_transform().

    # Returns a (n_regimes,) float32 array of per-class τ means (one scalar per
    # class, representing the mean τ averaged over all d_z dimensions), sorted in
    # ascending order (clear=0 has smallest τ, overcast=n-1 has largest τ).

    # Returns None if the centroids were not found in latent_tau_stats.json —
    # the caller falls back to null regime tokens in that case.

    # The centroids live in the same τ-space that the denoiser operates in
    # (after per-dim normalisation + log map by LatentTauTransform), so they
    # are geometrically correct — unlike the old K*-to-τ heuristic which
    # ignored per-dimension z_min/z_max scaling entirely (NEW-5).
    # 
    tau_stats_path = cfg.get("paths", {}).get("latent_tau_stats", None)
    if tau_stats_path is None or not Path(tau_stats_path).exists():
        log.warning(
            "_load_tau_class_centroids: latent_tau_stats not found at '%s'. "
            "Regime feedback loop disabled.", tau_stats_path,
        )
        return None

    try:
        import json as _json
        with open(tau_stats_path) as f:
            blob = _json.load(f)
        centroids = blob.get("class_tau_centroids", None)
        if centroids is None:
            log.warning(
                "_load_tau_class_centroids: 'class_tau_centroids' key missing from %s. "
                "Re-run fit_latent_tau_transform (train_diffusion) to write them.",
                tau_stats_path,
            )
            return None
        arr = np.array(centroids, dtype=np.float32)
        log.info(
            "Regime centroids (τ-space) loaded: %s",
            [round(float(v), 4) for v in arr],
        )
        return arr
    except Exception as e:
        log.warning("_load_tau_class_centroids: failed to load centroids: %s", e)
        return None


@torch.no_grad()
def _infer_regime_ids_from_tau(
    tau_hat0:          torch.Tensor,           # (1, W, d_z) predicted clean τ̂₀
    class_centroids:   torch.Tensor,           # (n_regimes,) per-class τ scalar centroids
    location_prior:    Optional[np.ndarray] = None,  # (n_regimes,) location-aware prior
    prior_strength:    float = 0.30,           # blend weight for location prior
    temperature:       float = 0.5,            # softmax temperature — read from config
) -> torch.Tensor:                             # (1, W) int64 regime labels
    # Map τ̂₀ predictions to 4-class regime labels via nearest centroid.

    # Per-day τ mean (mean over d_z) is compared against the n_regimes scalar
    # centroids fitted from the training latent cache.  The nearest centroid
    # (in L1) determines the regime label.

    # This is geometrically correct: centroids live in the same τ-space the
    # denoiser operates in (per-dim normalised + log-mapped), so the comparison
    # is on the right scale regardless of per-dimension z_min/z_max (fixes NEW-5).

    # The 4-class assignment (0=clear … 3=overcast) matches the RegimeEmbedding
    # vocab, so the denoiser always receives a valid label, including class 3.

    # Parameters
    # ----------
    # tau_hat0       : (1, W, d_z) — denoiser's τ̂₀ prediction at step k
    # class_centroids: (n_regimes,) — ascending-τ scalar centroids from training
    # temperature    : softmax temperature for centroid distance → probability.
    #                  Read from cfg["inference"]["regime_feedback_temperature"].
    #                  Lower = sharper (nearer to argmax); higher = more uniform.
    #                  0.5 is a reasonable default; tune if overcast is never sampled.

    # Returns
    # -------
    # (1, W) int64 tensor on the same device as tau_hat0.
    # 
    tau_mean = tau_hat0.mean(dim=2)                  # (1, W) — mean over d_z per day

    # Soft regime assignment via nearest-centroid distance → softmax probability.
    # Hard argmax creates a self-reinforcing clear-sky loop.
    # Temperature controls how sharp the assignment is:
    #   low T (e.g. 0.1) → near-argmax, strong self-reinforcement
    #   high T (e.g. 2.0) → uniform, no useful signal
    #   0.5 balances self-consistency with exploration of other regimes.
    # Tune via cfg["inference"]["regime_feedback_temperature"].
    cents     = class_centroids.to(tau_hat0.device).view(1, 1, -1)  # (1, 1, n_regimes)
    dists     = (tau_mean.unsqueeze(2) - cents).abs()               # (1, W, n_regimes)
    log_probs = -dists / (temperature + 1e-8)                       # (1, W, n_regimes)
    probs     = torch.softmax(log_probs, dim=2)                     # (1, W, n_regimes)

    # Blend with location prior if provided
    if location_prior is not None and prior_strength > 0.0:
        loc_prior_t = torch.from_numpy(location_prior).to(probs.device)
        loc_prior_t = loc_prior_t.view(1, 1, -1).expand_as(probs)
        probs = (1.0 - prior_strength) * probs + prior_strength * loc_prior_t
        probs = probs / probs.sum(dim=2, keepdim=True).clamp(min=1e-8)

    B, W_dim, R = probs.shape
    labels = torch.multinomial(
        probs.reshape(B * W_dim, R), num_samples=1
    ).reshape(B, W_dim).long()                                  # (1, W)
    return labels



def _utc_offset_from_profile(profile: Dict) -> Optional[float]:
    # Read the exact UTC offset (fractional hours) from a real profile's timestamps.

    # SolarPreprocessor anchors timestamps[0] at local midnight expressed in UTC,
    # so the offset is simply the fractional hours between UTC midnight and the
    # first timestamp.  Supports sub-hour offsets (e.g. India +5:30, Nepal +5:45).

    # Returns None if timestamps are missing or unparseable — caller falls back
    # to the rounded longitude estimate.
    # 
    try:
        import pandas as _pd
        ts0 = _pd.Timestamp(profile["timestamps"][0])   # tz-naive UTC wall-clock
        # ts0 is the UTC time that corresponds to 00:00 local.
        # UTC offset = -(ts0 - midnight_UTC) in hours, i.e. how many hours
        # *ahead* of UTC the station is.
        midnight_utc = _pd.Timestamp(profile["date"])   # 00:00 UTC
        offset_hours = -(ts0 - midnight_utc).total_seconds() / 3600.0
        return offset_hours
    except Exception:
        return None


def _compute_solar_geometry(
    dates: List[date],
    lat: float,
    lon: float,
    zenith_threshold: float,
    day_feat_stats,    # DayFeatureNormStats
    cfg: Dict,
    utc_offset_hours: Optional[float] = None,   # exact offset from real profile
) -> Tuple[List[Dict], np.ndarray, np.ndarray, int]:
    # Compute all physics conditioning tensors for a list of dates.

    # All inputs come from pvlib + lat/lon — no observations needed.

    # Timestamps are built in UTC anchored at local midnight.  When
    # ``utc_offset_hours`` is provided (read from a real profile's timestamps[0]
    # by ``_utc_offset_from_profile``), it is used directly so the generated
    # sunlit window aligns exactly with the preprocessed real profiles.  Without
    # it, the offset is estimated from longitude (1 h per 15°, rounded), which
    # can introduce a systematic 1–2 timestep shift for stations that don't sit
    # on a round-hour timezone meridian.

    # Returns
    # -------
    # profiles        : list of synthetic profile dicts (same format as dataset)
    # day_feat_arr    : (N, 7) normalised day-level features
    # location_enc    : (4,) sin/cos lat/lon encoding
    # n_steps_per_day : int — inferred from inference_steps_per_day in cfg
    # 
    import pandas as pd
    import pvlib
    from data.physics_utils import (
        extract_day_features,
        extract_location_features,
        extract_climate_features,
        N_CLIMATE_FEATURES,
    )

    n_steps  = int(cfg["data"].get("inference_steps_per_day", 144))
    freq_min = int(24 * 60 / n_steps)

    if utc_offset_hours is None:
        utc_offset_hours = round(lon / 15.0)
        log.warning(
            "_compute_solar_geometry: utc_offset_hours not provided — "
            "estimating from longitude (%.4f°) as %.1f h.  Pass a real "
            "profile's UTC offset to avoid timestep misalignment.",
            lon, utc_offset_hours,
        )
    else:
        log.info(
            "_compute_solar_geometry: using exact UTC offset %.4f h "
            "derived from real profile timestamps.", utc_offset_hours,
        )

    site     = pvlib.location.Location(lat, lon)
    profiles = []

    for d in dates:
        local_midnight_utc = pd.Timestamp(str(d), tz="UTC") - pd.Timedelta(hours=utc_offset_hours)
        ts = pd.date_range(start=local_midnight_utc, periods=n_steps,
                           freq=f"{freq_min}min", tz="UTC")

        sol = site.get_solarposition(ts)
        cs  = site.get_clearsky(ts, model="ineichen")
        ts_naive = ts.tz_localize(None)

        profile = {
            "date":       str(d),
            "timestamps": [str(t) for t in ts_naive],
            "solar_noon": str(ts_naive[int(sol["zenith"].values.argmin())]),
            "csi": np.ones(n_steps, dtype=np.float32).tolist(),
            "deterministic_geometry": {
                "solar_zenith_angle": sol["zenith"].values.tolist(),
                "ghi_clear_sky":      cs["ghi"].values.tolist(),
            },
            "lat":     lat,
            "lon":     lon,
            "station": "inference",
        }
        profiles.append(profile)

    # 7-dim solar day features (normalised)
    day_feat_arr = np.stack([
        day_feat_stats.normalize(extract_day_features(p, zenith_threshold))
        for p in profiles
    ])   # (N, 7)

    # Climate features: computed once from lat/lon, broadcast to all days.
    # Uses kgcpy lookup: Köppen zone → one-hot (6) + irradiance quantiles (4) = 10 dims.
    # Same call as in dataset.__getitem__ — inference is identical to training.
    # climate_stats normalises the irradiance quantile dims; loaded from config path.
    clim_raw = extract_climate_features(lat, lon)   # (N_CLIMATE_FEATURES=10,)
    climate_stats_path = cfg.get("paths", {}).get("norm_stats_climate", None)
    if climate_stats_path is not None and Path(climate_stats_path).exists():
        from data.physics_utils import ClimateFeatNormStats as _CFS
        _cs = _CFS()
        _cs.load(climate_stats_path)
        clim_norm = _cs.normalize(clim_raw)   # (10,)
        log.info("Climate stats loaded from %s for inference.", climate_stats_path)
    else:
        # Fallback: use raw features — Köppen one-hot already {0,1},
        # irradiance quantiles are large (Wh/m²/year) but the denoiser
        # will still learn from them; z-scoring just improves convergence.
        clim_norm = clim_raw
        if climate_stats_path is not None:
            log.warning(
                "Climate stats not found at %s — using raw climate features. "
                "Run fit_and_save_norm_stats to generate norm_stats_climate.json.",
                climate_stats_path,
            )

    # Append climate features to every day: (N, 7) → (N, 7+10=17)
    clim_broadcast = np.tile(clim_norm, (len(profiles), 1))      # (N, 10)
    day_feat_arr   = np.concatenate([day_feat_arr, clim_broadcast], axis=1)  # (N, 17)

    location_enc = extract_location_features(lat, lon)   # (4,)

    return profiles, day_feat_arr, location_enc, n_steps


def _build_intraday_tensor(
    profiles: List[Dict],
    zenith_threshold: float,
    intraday_stats,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray]]:
    # Build padded intraday_phys tensor and valid_mask for a window of profiles.

    # Returns
    # -------
    # intraday  : (W, T_max, 3) float32 on device — normalised [zenith, ETR, GCS]
    # valid_mask: (W, T_max) bool on device — True = real (sunlit) timestep
    # sunlit_idx: list of W arrays with sunlit timestep indices into the full day
    # 
    from data.physics_utils import extract_intraday_matrix, extract_sunlit_mask

    W          = len(profiles)
    matrices   = []
    masks_raw  = []
    sunlit_idx = []

    for p in profiles:
        mat = extract_intraday_matrix(p)             # (T, 3)
        sun = extract_sunlit_mask(p, zenith_threshold)  # (T,) bool
        matrices.append(intraday_stats.normalize(mat[sun]))  # (T_sun, 3)
        masks_raw.append(sun)
        sunlit_idx.append(np.where(sun)[0])

    T_max        = max(m.shape[0] for m in matrices)
    intraday_pad = np.zeros((W, T_max, 3), dtype=np.float32)
    mask_pad     = np.zeros((W, T_max),    dtype=bool)

    for i, m in enumerate(matrices):
        t = m.shape[0]
        intraday_pad[i, :t, :] = m
        mask_pad[i, :t]        = True

    return (
        torch.from_numpy(intraday_pad).to(device),
        torch.from_numpy(mask_pad).to(device),
        sunlit_idx,
    )


# 
# Core reverse sampler — DDIM (fast, default) or DDPM (stochastic)
# 

@torch.no_grad()
def _ddpm_reverse(
    denoiser: SolarDenoiser,
    schedule: TauNoiseSchedule,
    day_feat: torch.Tensor,       # (1, W, 7)
    location: torch.Tensor,       # (1, 4)
    intraday: torch.Tensor,       # (1, W, T_max, 3)
    valid_mask: torch.Tensor,     # (1, W, T_max)
    d_z: int,
    guidance_scale: float,
    context_tau: Optional[torch.Tensor] = None,  # (1, n_ctx, d_z) from prev window
    n_ctx: int = 0,
    context_noise_std: float = 0.0,  # unused — kept for API compatibility
    ddim_steps: Optional[int] = None,  # None → DDPM (all T steps); int → DDIM
    ddim_eta:   float = 0.0,           # DDIM stochasticity (0=deterministic)
    regime_ids: Optional[torch.Tensor] = None,            # (1, W) int64 — initial labels (overridden by feedback)
    tau_class_centroids: Optional[torch.Tensor] = None,   # (n_regimes,) — 4-class nearest-centroid thresholds
    location_prior: Optional[np.ndarray] = None,          # (n_regimes,) location-aware regime prior
    prior_strength: float = 0.30,                         # blend weight for location prior
    feedback_temperature: float = 0.5,                    # softmax temperature for centroid→label; read from cfg
    tau_max:    float = 4.605,         # −log(z_norm_clamp_lo); physical upper bound on τ̂₀
) -> torch.Tensor:                 # (1, W, d_z) in τ-space
    # Reverse diffusion producing τ latents for a window of W days.

    # Sampler selection
    # -----------------
    # ddim_steps=None  : full DDPM (schedule.T steps, stochastic).
    # ddim_steps=N     : DDIM with N evenly-spaced steps (default 50).
    #                    eta=0 → deterministic; eta=1 → DDPM-like noise.

    # Context pinning (autoregressive generation / RePaint)
    # -----------------------------------------------------
    # The first n_ctx positions are hard-pinned at every denoising step:
    #     τ[:, :n_ctx, :] = √ᾱ_k · context_tau + √(1−ᾱ_k) · fresh_noise

    # Self-consistent 4-class regime feedback loop
    # --------------------------------------------
    # If tau_class_centroids (n_regimes,) is provided, regime labels are updated
    # at every denoising step via nearest-centroid assignment in τ-space:

    #   1. Run a single forward pass (without CFG) to get τ̂₀ estimate.
    #   2. Compute per-day τ mean (mean over d_z dimensions).
    #   3. Assign the nearest of the n_regimes scalar centroids → hard label.
    #   4. Pass those labels as regime_ids to the full CFG forward pass.

    # Centroids come from class_tau_centroids in latent_tau_stats.json, fitted
    # from the training latent cache — geometrically correct in τ-space (fixes
    # the K*-to-τ heuristic that ignored per-dim z_min/z_max scaling).

    # Returns τ-space latents. Caller applies LatentTauTransform.from_tau_space()
    # then vae.decode().
    # 
    device = day_feat.device
    W      = day_feat.shape[1]

    tau = torch.randn(1, W, d_z, device=device)

    has_context  = context_tau is not None and n_ctx > 0
    use_feedback = tau_class_centroids is not None

    # Current regime labels — updated every step when use_feedback is True.
    # When feedback is off, these stay fixed (or None → null tokens).
    current_regime_ids = regime_ids   # (1, W) int64 or None

    denoiser.eval()

    # Pre-compute intraday physics keys/values once — deterministic across steps.
    phys_kv_cache, key_pad_cache = denoiser.precompute_intraday_kv(
        intraday, valid_mask
    )

    # Build step sequence: DDIM (fast) or DDPM (full T steps)
    use_ddim = ddim_steps is not None
    if use_ddim:
        step_seq = schedule.build_ddim_steps(ddim_steps)   # descending, ends at 0
    else:
        step_seq = list(reversed(range(schedule.T)))        # DDPM: T-1 … 0

    for seq_idx, k in enumerate(step_seq):
        k_tensor = torch.full((1,), k, dtype=torch.long, device=device)

        # ── Hard context pinning (RePaint) ────────────────────────────────────
        if has_context:
            ab_k      = schedule.alpha_bars[k]
            eps_ctx   = torch.randn_like(context_tau)
            tau[:, :n_ctx, :] = (ab_k.sqrt() * context_tau
                                  + (1.0 - ab_k).sqrt() * eps_ctx)

        # ── Self-consistent 4-class regime feedback ───────────────────────────
        # Cheap single forward pass (no CFG) to predict τ̂₀, then assign regime
        # labels by nearest centroid in τ-space.  Skipped at k=0 (final step).
        if use_feedback and k > 0:
            no_drop   = torch.zeros(1, device=device, dtype=torch.bool)
            v_cheap   = denoiser.forward(
                tau, k_tensor, day_feat, location,
                intraday, valid_mask, drop_mask=no_drop,
                phys_kv_cache=phys_kv_cache,
                key_pad_cache=key_pad_cache,
                regime_ids=current_regime_ids,
            )
            # Recover τ̂₀ from v-prediction: τ̂₀ = √ᾱ_k·τₖ − √(1−ᾱ_k)·v̂
            sab_k    = schedule.sqrt_alpha_bars[k]
            som_k    = schedule.sqrt_one_minus[k]
            tau_hat0 = sab_k * tau - som_k * v_cheap   # (1, W, d_z)
            current_regime_ids = _infer_regime_ids_from_tau(
                tau_hat0, tau_class_centroids,
                location_prior=location_prior,
                prior_strength=prior_strength,
                temperature=feedback_temperature,
            )   # (1, W) int64 — updated for this step's CFG call

        # ── Full CFG forward pass with up-to-date regime labels ───────────────
        v_pred = denoiser.forward_cfg(
            tau, k_tensor, day_feat, location,
            intraday, valid_mask, guidance_scale,
            phys_kv_cache=phys_kv_cache,
            key_pad_cache=key_pad_cache,
            regime_ids=current_regime_ids,
        )

        if use_ddim:
            k_prev = step_seq[seq_idx + 1] if seq_idx + 1 < len(step_seq) else -1
            tau = schedule.p_sample_ddim(tau, k, k_prev, v_pred, eta=ddim_eta,
                                         tau_max=tau_max)
        else:
            tau = schedule.p_sample(tau, k, v_pred, tau_max=tau_max)

    return tau   # (1, W, d_z)


# 
# Post-processing
# 

def _postprocess_day(
    k_sun: np.ndarray,       # (T_sun,) reconstructed sunlit K*
    sunlit_idx: np.ndarray,  # indices into the full T_full-length day
    T_full: int,             # total timesteps per day
    k_max: float,            # from cfg["physics"]["k_max"]
    smooth_sigma: float,     # from cfg["inference"]["smooth_sigma"]
    boundary_width: int,     # from cfg["inference"]["boundary_width"]
) -> np.ndarray:             # (T_full,) full-day K* profile
    # Build full-day K* profile from sunlit reconstruction.

    # Steps:
    #   1. Clip to [0, k_max]         — physical bound
    #   2. Insert into full-day array — night positions remain 0
    #   3. Gaussian smooth sunrise/sunset boundaries only — removes hard edges
    #   4. Hard-enforce night = 0.0   — physical constraint overrides smoothing
    # 
    profile = np.zeros(T_full, dtype=np.float32)

    k_sun           = np.clip(k_sun, 0.0, k_max)
    profile[sunlit_idx] = k_sun

    if len(sunlit_idx) == 0:
        return profile

    if smooth_sigma > 0 and boundary_width > 0:
        # Sunrise boundary
        sr_s = max(0, sunlit_idx[0] - boundary_width)
        sr_e = min(T_full, sunlit_idx[0] + boundary_width + 1)
        profile[sr_s:sr_e] = gaussian_filter1d(profile[sr_s:sr_e], sigma=smooth_sigma)

        # Sunset boundary
        ss_s = max(0, sunlit_idx[-1] - boundary_width)
        ss_e = min(T_full, sunlit_idx[-1] + boundary_width + 1)
        profile[ss_s:ss_e] = gaussian_filter1d(profile[ss_s:ss_e], sigma=smooth_sigma)

    # Hard-enforce night = 0 (overrides any smoothing bleed)
    night          = np.ones(T_full, dtype=bool)
    night[sunlit_idx] = False
    profile[night] = 0.0

    return profile



# 
# Main generator
# CSV output helper
# 

def _save_timeseries_csv(
    path: str,
    dates: List[str],
    generated: np.ndarray,                      # (N, T)
    real: Optional[np.ndarray] = None,          # (N, T) or None
    clearsky_ghi: Optional[np.ndarray] = None,  # (N, T) W/m2 -- pvlib, same grid
    zenith: Optional[np.ndarray] = None,        # (N, T) degrees -- pvlib, same grid
) -> None:
    # Save timeseries to a CSV with one row per timestep.

    # Columns: ``date, timestep, generated[, real][, clearsky_ghi, zenith]``

    # ``clearsky_ghi`` and ``zenith`` are derived from the same pvlib timestamp
    # grid used during generation, so they are guaranteed aligned with
    # ``generated`` index-for-index.

    # ``real`` column is omitted when not provided (standalone generation).
    # Gap days in ``real`` (all-zero rows) are written as-is.
    # 
    import pandas as pd
    from pathlib import Path as _Path

    N, T = generated.shape
    records = []
    for i, date_str in enumerate(dates):
        for t in range(T):
            row: Dict = {"date": date_str, "timestep": t, "generated": generated[i, t]}
            if real is not None:
                row["real"] = real[i, t]
            if clearsky_ghi is not None:
                row["clearsky_ghi"] = float(clearsky_ghi[i, t])
            if zenith is not None:
                row["zenith"] = float(zenith[i, t])
            records.append(row)

    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    log.info("Timeseries saved -> %s  (%d rows)", path, len(records))


@torch.no_grad()
def generate_sequence(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    vae: SolarVAE,
    denoiser: SolarDenoiser,
    schedule: TauNoiseSchedule,
    latent_transform: LatentTauTransform,
    cfg: Dict,
    intraday_stats,
    day_feat_stats,
    device: torch.device,
    utc_offset_hours: Optional[float] = None,  # exact UTC offset from real profile
) -> np.ndarray:
    # Generate a K* profile sequence for an arbitrary date range.

    # The only inputs required from the user are start_date, end_date, lat, lon.
    # All conditioning tensors are computed deterministically from pvlib.

    # Parameters
    # ----------
    # start_date / end_date : 'YYYY-MM-DD' strings (inclusive)
    # lat, lon              : location in decimal degrees
    # vae                   : trained AE (eval mode, frozen)
    # denoiser              : trained diffusion model (eval mode)
    # schedule              : TauNoiseSchedule (fitted at training time)
    # latent_transform      : fitted LatentTauTransform (tau <-> z mapping)
    # cfg                   : full config dict
    # intraday_stats        : fitted IntraDayNormStats
    # day_feat_stats        : fitted DayFeatureNormStats
    # device                : torch device
    # utc_offset_hours      : exact UTC offset (fractional hours) read from the
    #                         first real profile's timestamps[0] by
    #                         ``_utc_offset_from_profile``.  When provided, pvlib
    #                         timestamps are anchored at exactly the same UTC point
    #                         as the real profiles, eliminating the timestep shift.
    #                         Falls back to round(lon/15) when None.

    # CSV output
    # ----------
    # If ``cfg["paths"]["output_csv"]`` is set, the generated timeseries is saved
    # to that path with one row per timestep: columns
    # ``date, timestep, generated[, real], clearsky_ghi, zenith``.
    # ``clearsky_ghi`` and ``zenith`` come from the same pvlib grid as
    # ``generated`` and are therefore guaranteed to be aligned with it
    # index-for-index (same pvlib timestamp anchor, same n_steps, same UTC offset).

    # Returns
    # -------
    # output       : (N_days, T_full) float32 -- K* profiles, 0.0 at night/twilight
    # clearsky_ghi : (N_days, T_full) float32 -- pvlib Ineichen GHI W/m2
    # zenith       : (N_days, T_full) float32 -- solar zenith angle degrees
    # 
    dc       = cfg["diffusion"]
    ic       = cfg["inference"]
    pc       = cfg["physics"]
    W        = cfg["data"]["window_size"]

    # NOTE on bypass_alpha: do NOT zero it at inference.
    # The AEDecoder bypass lane is initialised at -3.0 (sigmoid≈0.047, nearly off)
    # so k_dec learns the full profile shape first. bypass_alpha only grows during
    # training if it genuinely reduces loss. Zeroing it at inference would remove
    # a contribution that k_dec was jointly trained to expect, producing a
    # different decoder mode than what was trained. The -3.0 init in vae.py is
    # the correct fix for the flat-floor artifact — not inference-time zeroing.
    # Log the actual learned alpha so we can monitor it.
    with torch.no_grad():
        for _m in vae.modules():
            if hasattr(_m, "bypass_alpha"):
                learned_alpha = float(torch.sigmoid(_m.bypass_alpha).item())
                log.info(
                    "bypass_alpha at inference: sigmoid(%.4f) = %.4f  "
                    "(0.047=near-off, 0.5=half-strength; retrain VAE if > 0.15 "
                    "and flat-floor artifacts appear — means old 0.0 init was used).",
                    float(_m.bypass_alpha.item()), learned_alpha,
                )
    S        = ic["stride"]      # new days generated per sliding window step
    n_ctx    = W - S             # context days carried over from previous window
    d_z      = cfg["vae"]["latent_dim"] + int(cfg["vae"].get("z_var_dim", 0))
    k_max    = pc["k_max"]
    z_thr    = cfg["data"]["zenith_threshold_deg"]
    guidance = dc["cfg_guidance_scale"]
    ctx_noise_std = float(ic.get("context_noise_std", 0.0))   # kept for compat; no effect

    # DDIM: read from inference config. ddim_steps=None falls back to full DDPM.
    # 50 steps is the recommended default (>95% quality at 1/20th DDPM cost).
    ddim_steps = ic.get("ddim_steps", 50)
    if ddim_steps is not None:
        ddim_steps = int(ddim_steps)
    ddim_eta   = float(ic.get("ddim_eta", 0.0))

    # τ_max = −log(z_norm_clamp_lo): physical upper bound for τ̂₀ clamping in samplers.
    # Matches the LatentTauTransform lower clamp on z_norm (to_tau_space).
    import math as _math
    _z_norm_clamp_lo = float(cfg["vae"].get("z_norm_clamp_lo", 0.01))
    tau_max_gen = float(-_math.log(_z_norm_clamp_lo))

    # Build date list
    d0    = date.fromisoformat(start_date)
    d1    = date.fromisoformat(end_date)
    dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    N     = len(dates)

    # Compute all deterministic solar geometry from pvlib.
    # Pass the exact UTC offset so generated timestamps align with real profiles.
    profiles, day_feat_arr, location_enc, n_steps = _compute_solar_geometry(
        dates, lat, lon, z_thr, day_feat_stats, cfg,
        utc_offset_hours=utc_offset_hours,
    )

    # Extract clearsky GHI and zenith from profiles (same pvlib grid as output).
    # Shape: (N, T_full) — aligned index-for-index with the generated K* array.
    all_clearsky_ghi = np.array([
        p["deterministic_geometry"]["ghi_clear_sky"] for p in profiles
    ], dtype=np.float32)   # (N, T_full)
    all_zenith = np.array([
        p["deterministic_geometry"]["solar_zenith_angle"] for p in profiles
    ], dtype=np.float32)   # (N, T_full)
    T_full     = n_steps
    location_t = torch.from_numpy(location_enc).unsqueeze(0).to(device)  # (1, 4)

    #  4-class regime conditioning — self-consistent nearest-centroid feedback 
    # Load per-class τ centroids from latent_tau_stats.json (written by
    # fit_latent_tau_transform).  These are geometrically correct in the same
    # τ-space the denoiser operates in — fixes the old K*-to-τ heuristic that
    # ignored per-dim z_min/z_max (Problem NEW-5).
    # If centroids are unavailable, the feedback loop falls back to null regime
    # tokens (regime conditioning disabled) rather than using wrong thresholds.
    tau_centroids_np = _load_tau_class_centroids(latent_transform, cfg)
    if tau_centroids_np is not None:
        tau_class_centroids = torch.from_numpy(tau_centroids_np).to(device)
        log.info(
            "4-class regime feedback loop enabled with %d centroids: %s",
            len(tau_centroids_np),
            [round(float(v), 4) for v in tau_centroids_np],
        )
    else:
        tau_class_centroids = None
        log.warning(
            "Regime feedback loop disabled: class_tau_centroids not found. "
            "Regime conditioning uses null tokens throughout generation."
        )

    #  Regime feedback temperature 
    # Controls sharpness of centroid → label assignment.
    # Lower = nearer to hard argmax (self-reinforcing); higher = more uniform.
    # Default 0.5 balances self-consistency with regime diversity.
    _feedback_temp = float(ic.get("regime_feedback_temperature", 0.5))

    # Location-aware regime prior is DISABLED.
    # The prior biases regime sampling toward the climatological expectation
    # at the target site, but in practice it over-constrains generation:
    # high-latitude sites (e.g. Sweden) get a low p_clear prior that
    # self-reinforces through the feedback loop, suppressing clear days entirely.
    # The tau-centroid feedback alone is sufficient — the denoiser's own
    # tau predictions reflect what regime the chain is actually heading toward.
    # The prior can be re-enabled after retraining with more climatologically
    # diverse stations, at which point the model will have learned the correct
    # regime frequencies for different climates rather than being forced by a
    # pvlib-derived heuristic.
    # [next train] Re-evaluate regime_prior_strength after adding cloudy-climate
    # stations. Start at 0.10 and only raise if generation is still too clear-sky.
    _prior_strength  = 0.0
    _location_prior  = None
    log.info(
        "Location regime prior disabled — using tau-centroid feedback only "
        "(lat=%.2f lon=%.2f). Re-enable after retraining with diverse stations.",
        lat, lon,
    )

    # Allocate tau-space output buffer
    all_tau = torch.zeros(N, d_z, device=device)

    if N <= W:
        # Single forward pass -- all days fit in one window
        intraday, vmask, _ = _build_intraday_tensor(profiles[:N], z_thr, intraday_stats, device)
        df = torch.from_numpy(day_feat_arr[:N]).unsqueeze(0).to(device)   # (1, N, 7)

        tau_out = _ddpm_reverse(
            denoiser, schedule, df, location_t,
            intraday.unsqueeze(0), vmask.unsqueeze(0),
            d_z, guidance,
            context_noise_std=ctx_noise_std,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            regime_ids=None,
            tau_class_centroids=tau_class_centroids,
            location_prior=_location_prior,
            prior_strength=_prior_strength,
            feedback_temperature=_feedback_temp,
            tau_max=tau_max_gen,
        )   # (1, N, d_z) in tau-space
        all_tau[:N] = tau_out[0]

    else:
        # Autoregressive sliding window
        context_tau: Optional[torch.Tensor] = None
        window_starts = list(range(0, N - W + 1, S))
        if window_starts and window_starts[-1] + W < N:
            window_starts.append(N - W)

        for win_idx, win_start in enumerate(window_starts):
            win_end   = min(win_start + W, N)
            win_profs = profiles[win_start:win_end]
            win_len   = win_end - win_start

            intraday, vmask, _ = _build_intraday_tensor(
                win_profs, z_thr, intraday_stats, device
            )
            df = torch.from_numpy(
                day_feat_arr[win_start:win_end]
            ).unsqueeze(0).to(device)   # (1, win_len, 7)

            curr_n_ctx = n_ctx if (context_tau is not None and win_idx > 0) else 0

            tau_out = _ddpm_reverse(
                denoiser, schedule, df, location_t,
                intraday.unsqueeze(0), vmask.unsqueeze(0),
                d_z, guidance,
                context_tau=context_tau,
                n_ctx=curr_n_ctx,
                context_noise_std=ctx_noise_std,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                regime_ids=None,
                tau_class_centroids=tau_class_centroids,
                location_prior=_location_prior,
                prior_strength=_prior_strength,
                feedback_temperature=_feedback_temp,
                tau_max=tau_max_gen,
            )   # (1, win_len, d_z)

            # Store only the newly generated positions (not context)
            new_start = win_start + curr_n_ctx
            new_tau   = tau_out[0, curr_n_ctx:]
            n_new     = new_tau.shape[0]
            all_tau[new_start: new_start + n_new] = new_tau

            # Last n_ctx tau latents become context for the next window.
            # FIX G4: Add small Gaussian noise before passing as context.
            # Without perturbation, context_tau carries the clear-sky signal
            # rigidly into every window, making cloudy spells impossible across
            # window boundaries. noise_std=0.3 is small relative to tau_std≈1
            # but enough to allow the reverse chain to explore other regimes.
            # Read from config: inference.context_noise_std (default 0.0 = off).
            # Pass context tau cleanly — no artificial perturbation.
            # The context carries learned solar geometry conditioning.
            context_tau = tau_out[:, -n_ctx:, :].clone()

    # Convert tau → z → K* profiles
    vae.eval()
    output = np.zeros((N, T_full), dtype=np.float32)

    _d_z_flat_gen = int(cfg["vae"]["latent_dim"])         # z_flat dims only
    _d_z_var_gen  = int(cfg["vae"].get("z_var_dim", 0))  # z_var dims

    for i, (profile, _) in enumerate(zip(profiles, dates)):
        tau_i      = all_tau[i].unsqueeze(0)              # (1, d_z_full)
        # Only invert z_flat dims — z_var dims have a ~4× narrower range in
        # tau-space and would saturate from_tau_space().  The decoder slices
        # z[:, :latent_dim] internally so z_var is already ignored.
        tau_i_flat = tau_i[:, :_d_z_flat_gen]            # (1, latent_dim)
        z_flat_i   = latent_transform.from_tau_space(tau_i_flat)
        if _d_z_var_gen > 0:
            z_var_zeros = torch.zeros(
                1, _d_z_var_gen, device=tau_i.device, dtype=tau_i.dtype
            )
            z_i = torch.cat([z_flat_i, z_var_zeros], dim=-1)
        else:
            z_i = z_flat_i

        intraday_i, vmask_i, sunlit_idx_list = _build_intraday_tensor(
            [profile], z_thr, intraday_stats, device
        )
        sunlit_idx = sunlit_idx_list[0]
        T_sun      = int(vmask_i[0].sum().item())

        if T_sun == 0:
            output[i] = 0.0   # polar night
            continue

        phys_i = intraday_i[0, :T_sun].unsqueeze(0)   # (1, T_sun, phys_dim)
        with torch.no_grad():
            k_sun = vae.decode(z_i, phys_i, T_sun)    # (1, T_sun)
        k_sun = k_sun[0].cpu().numpy()

        output[i] = _postprocess_day(
            k_sun, sunlit_idx, T_full, k_max,
            ic["smooth_sigma"], ic["boundary_width"],
        )

    log.info("Generated %d days  shape=%s", N, output.shape)

    csv_path = cfg.get("paths", {}).get("output_csv", None)
    if csv_path is not None:
        _save_timeseries_csv(
            csv_path,
            dates        = [str(d0 + timedelta(days=i)) for i in range(N)],
            generated    = output,
            clearsky_ghi = all_clearsky_ghi,
            zenith       = all_zenith,
        )

    return output, all_clearsky_ghi, all_zenith


# 
# Test-set loader — extract real K* from a pre-built profile JSON
# 

def load_test_profiles(test_profiles_path: str) -> List[Dict]:
    # Load test profiles from the JSON file written by SolarPreprocessor.

    # Returns the raw list of profile dicts sorted by date, exactly as
    # SolarSequenceDataset would see them.
    # 
    import json
    with open(test_profiles_path) as f:
        profiles = json.load(f)
    profiles = sorted(profiles, key=lambda p: p["date"])
    log.info("Loaded %d test profiles  date_range=%s – %s",
             len(profiles), profiles[0]["date"], profiles[-1]["date"])
    return profiles


def test_set_date_range(profiles: List[Dict]) -> Tuple[str, str]:
    # Return (first_date, last_date) of the test profiles as 'YYYY-MM-DD' strings
    dates = sorted(p["date"] for p in profiles)
    return dates[0], dates[-1]


def extract_real_sequence(
    profiles: List[Dict],
    start_date: str,
    end_date: str,
    cfg: Dict,
    intraday_stats,   # IntraDayNormStats -- used only to read zenith_threshold
) -> Tuple[np.ndarray, List[str]]:
    # Extract the real K* (N, T_full) array from test profiles for a date range.

    # Gap-tolerant: if the test profiles have missing days within the requested
    # date range (station outages, QC failures, etc.) those days are filled with
    # zeros in the real array and flagged as 'gap' in the returned dates list.
    # The returned array always has exactly one row per calendar day in the range,
    # so it aligns index-for-index with the generated sequence from
    # generate_sequence() which also covers every calendar day.

    # Parameters
    # ----------
    # profiles    : sorted list of test profile dicts
    # start_date  : 'YYYY-MM-DD' (inclusive)
    # end_date    : 'YYYY-MM-DD' (inclusive)
    # cfg         : full config dict — reads zenith_threshold, inference_steps_per_day
    # intraday_stats : only used to resolve zenith_threshold; not applied to real data

    # Returns
    # -------
    # real        : (N, T_full) float32  — K* with night=0.0; zeros for gap days
    # dates_used  : list of 'YYYY-MM-DD' strings in order (all calendar days)
    # 
    from data.physics_utils import extract_sunlit_mask

    z_thr  = cfg["data"]["zenith_threshold_deg"]
    T_full = int(cfg["data"].get("inference_steps_per_day", 144))

    # Index profiles by date for fast lookup
    by_date = {p["date"]: p for p in profiles}

    d0    = date.fromisoformat(start_date)
    d1    = date.fromisoformat(end_date)
    dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    N     = len(dates)

    # Identify gaps up-front and warn — do NOT raise
    missing = [str(d) for d in dates if str(d) not in by_date]
    if missing:
        log.warning(
            "extract_real_sequence: %d gap day(s) in test profiles within "
            "[%s, %s] — filling with zeros. Gaps: %s%s",
            len(missing), start_date, end_date,
            missing[:5],
            f" ... (+{len(missing)-5} more)" if len(missing) > 5 else "",
        )

    real = np.zeros((N, T_full), dtype=np.float32)
    used = []

    for i, d in enumerate(dates):
        ds = str(d)
        used.append(ds)

        if ds not in by_date:
            # Gap day — row stays all-zeros; already zeroed above
            continue

        p      = by_date[ds]
        csi    = np.array(p["csi"], dtype=np.float32)   # full-day K* from preprocessor
        T_prof = len(csi)

        # Resize to T_full if the profile has different temporal resolution.
        # Nearest-neighbour — preserves zeros at night without interpolation artefacts.
        if T_prof != T_full:
            idx     = (np.arange(T_full) * T_prof / T_full).astype(int)
            csi_out = csi[idx]
        else:
            csi_out = csi.copy()

        # Enforce physical night=0 using the same zenith threshold used during training
        sun_mask = extract_sunlit_mask(p, z_thr)   # (T_prof,) bool
        if T_prof != T_full:
            sun_out = sun_mask[(np.arange(T_full) * T_prof / T_full).astype(int)]
        else:
            sun_out = sun_mask

        csi_out[~sun_out] = 0.0
        real[i] = np.clip(csi_out, 0.0, float(cfg["physics"]["k_max"]))

    n_gap   = len(missing)
    n_valid = N - n_gap
    log.info(
        "Extracted real sequence  shape=%s  dates=%s – %s  "
        "valid=%d  gap_days=%d",
        real.shape, used[0], used[-1], n_valid, n_gap,
    )
    return real, used


def generate_for_test_period(
    start_date: str,
    end_date: str,
    profiles: List[Dict],
    vae: "SolarVAE",
    denoiser: "SolarDenoiser",
    schedule: "TauNoiseSchedule",
    latent_transform: "LatentTauTransform",
    cfg: Dict,
    intraday_stats,
    day_feat_stats,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Generate a sequence for a test-set date range AND extract the matching real data.

    # Gap-tolerant: if the test profiles have missing days within the requested
    # range, those days are filled with zeros in the real array.  The generated
    # array covers every calendar day regardless of gaps in the observed data,
    # since generation only needs pvlib solar geometry (no observations).

    # Both arrays have exactly one row per calendar day so they are index-aligned.
    # The caller can identify gap days by checking where real[i].sum() == 0 while
    # the calendar date lies within the expected sunlit season.

    # Parameters
    # ----------
    # start_date / end_date : 'YYYY-MM-DD' inclusive
    # profiles              : test profiles loaded via load_test_profiles()
    # all other args        : same as generate_sequence()

    # CSV output
    # ----------
    # If ``cfg["paths"]["output_csv"]`` is set, both timeseries are saved to that
    # path with one row per timestep: columns ``date, timestep, generated, real``.

    # Returns
    # -------
    # generated    : (N, T_full) float32
    # real         : (N, T_full) float32  — zeros for gap days
    # dates        : list[str] of YYYY-MM-DD, one per calendar day in range
    # clearsky_ghi : (N, T_full) float32  — pvlib Ineichen GHI W/m2, aligned to generated
    # zenith       : (N, T_full) float32  — solar zenith degrees, aligned to generated
    
    
    # Extract real data — gap-tolerant (no longer raises on missing days)
    real, dates_used = extract_real_sequence(
        profiles, start_date, end_date, cfg, intraday_stats
    )

    # Use lat/lon from the first AVAILABLE profile in the range
    by_date = {p["date"]: p for p in profiles}
    # Find first date that actually has a profile
    p0 = None
    for ds in dates_used:
        if ds in by_date:
            p0 = by_date[ds]
            break
    if p0 is None:
        raise ValueError(
            f"No test profiles found at all in range {start_date} – {end_date}. "
            "Cannot resolve station lat/lon for generation."
        )

    # Resolve lat/lon safely — explicit if/else avoids always-evaluate bug in
    # dict.get(key, _STATION_LATLON_FALLBACK(...)) where the fallback raises
    # even when the key is present (Python evaluates default args unconditionally).
    lat_raw = p0.get("lat", None)
    lon_raw = p0.get("lon", None)
    lat = float(lat_raw) if lat_raw is not None else float(_STATION_LATLON_FALLBACK(p0, "lat"))
    lon = float(lon_raw) if lon_raw is not None else float(_STATION_LATLON_FALLBACK(p0, "lon"))

    # Derive exact UTC offset from the first real profile's timestamps.
    # This is the key fix: SolarPreprocessor anchors timestamps[0] at local
    # midnight in UTC; reading it directly eliminates the round(lon/15) rounding
    # error that caused the 4-5 timestep shift in generated profiles.
    utc_offset = _utc_offset_from_profile(p0)
    if utc_offset is not None:
        log.info(
            "UTC offset read from profile timestamps: %.4f h  "
            "(station=%s  lon=%.4f  rounded_estimate=%.1f h)",
            utc_offset, p0.get("station", "?"), lon, round(lon / 15.0),
        )
    else:
        log.warning(
            "Could not read UTC offset from profile timestamps — "
            "falling back to round(lon/15)=%.1f h.  Timestep alignment "
            "may be off by a few steps.", round(lon / 15.0),
        )

    log.info("Test-set comparison  station=%s  lat=%.4f  lon=%.4f  dates=%s – %s",
             p0.get("station", "?"), lat, lon, dates_used[0], dates_used[-1])

    # Generate covers every calendar day in the range — pvlib needs no observations.
    # utc_offset_hours ensures the pvlib timestamp grid aligns with real profiles.
    # generate_sequence returns the physics arrays (clearsky_ghi, zenith) built
    # from the SAME pvlib call — no need to recompute, guaranteed aligned.
    # Location-aware regime prior is computed automatically inside generate_sequence
    # from pvlib GCS statistics — works for any lat/lon, no profiles needed.
    generated, all_clearsky_ghi, all_zenith = generate_sequence(
        start_date, end_date, lat, lon,
        vae, denoiser, schedule, latent_transform,
        cfg, intraday_stats, day_feat_stats, device,
        utc_offset_hours=utc_offset,
    )

    csv_path = cfg.get("paths", {}).get("output_csv", None)
    if csv_path is not None:
        _save_timeseries_csv(
            csv_path,
            dates        = dates_used,
            generated    = generated,
            real         = real,
            clearsky_ghi = all_clearsky_ghi,
            zenith       = all_zenith,
        )

    return generated, real, dates_used, all_clearsky_ghi, all_zenith


def _STATION_LATLON_FALLBACK(profile: Dict, coord: str = "lat") -> float:
    #Resolve lat or lon from profile, falling back to the dataset station table
    # Import lazily to avoid circular dependency
    from data.dataset import STATION_LATLON
    st = str(profile.get("station", "")).lower().strip()
    if coord == "lat":
        val = profile.get("lat", None)
        if val is not None:
            return float(val)
        if st in STATION_LATLON:
            return STATION_LATLON[st][0]
    else:
        val = profile.get("lon", None)
        if val is not None:
            return float(val)
        if st in STATION_LATLON:
            return STATION_LATLON[st][1]
    raise ValueError(
        f"Cannot resolve {coord} for station '{st}'. "
        "Add lat/lon to profile or add station to STATION_LATLON in dataset.py."
    )