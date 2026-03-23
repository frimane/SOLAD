"""
solar_diffusion/vae.py
-----------------------
Stage 1: Physics-conditioned deterministic 1-D Convolutional Autoencoder.

Encoder  (B, T_sun, 2+3+N_obs) → z_full (B, d_z + z_var_dim)
    Input channels: [K*, ΔK*, zenith_norm, ETR_norm, GCS_norm, obs...]
    ΔK* = temporal difference of K* (ramp signal at every encoder depth).
    N × _ConvBlock1d(stride, GroupNorm, per-timestep FiLM(physics), GELU)
    → key-query attention pool
    → Dropout → Linear(d_z)           → z_flat  (B, d_z)
    → _ZVarHead(k_star, valid_mask)   → z_var   (B, z_var_dim)   [normalised intraday std]
    → z_full = cat(z_flat, z_var)     (B, d_z + z_var_dim)

    z_var encodes intraday variability character (std, ramp_rate, etc.) computed
    directly from K* — not from conv features.  This captures the clear/overcast
    distinction with a physics-grounded signal that is cheap to compute and
    interpretable.

    GENERATION-TIME CONTRACT:
        The diffusion model learns to generate z_full jointly.
        The decoder receives ONLY z_flat = z_full[:, :d_z] (sliced, no extra params).
        The encoder is ABSENT at generation time — no mismatch possible.
        z_var dims are diffusion-model inputs only; they never enter the decoder.

Decoder  z_flat (B, d_z) + physics (B, T_sun, 3+2) → K̂* (B, T_sun) ∈ (0, k_max]
    Linear(d_z → ch[0]*init_len)
    → N × ConvTranspose1d(stride=2) + GroupNorm + FiLM(physics, masked mean)
                                     + z-FiLM(z_flat global)
                                     + PhysicsSkip(intraday_phys per-timestep)
                                     + GELU
    → interpolate(target_len)
    → cat(per-timestep physics+Δphysics, z_proj broadcast over T)
    → depthwise Conv1d(k=3) → pointwise Conv1d → GELU → Conv1d(1)
    → sigmoid · k_max  +  α · physics_bypass(GCS_mean)

Physics-only skip connections inject intraday_phys at every decoder depth.
No encoder feature maps used — fully generation-safe.

Decoder improvements vs original (all generation-safe):
  1. z-FiLM at every decoder block   — z_flat re-injected at each depth via γ(z),β(z)
                                        zero-init → identity at startup
  2. Δphysics input channels         — Δzenith, ΔETR added to decoder physics
                                        captures solar geometry ramp structure
  3. Depthwise-separable output head — local (k=3 depthwise) + global (pointwise)
                                        better intraday shape capacity same params
  4. Physics bypass lane             — learned α·GCS_baseline added to output
                                        escape hatch for uninformative z (overcast)
                                        α init=0 so no impact at startup
  6. Physics-only skip connections   — intraday_phys projected to each decoder channel
                                        width and injected as residual addition after
                                        each deconv block. Gives the decoder fine-grained
                                        per-timestep physics signal at every resolution
                                        level, not just the bottleneck via FiLM.
                                        GENERATION-SAFE: skips use only intraday_phys
                                        (deterministic solar geometry), never K* or obs.
                                        Controlled by cfg["vae"]["decoder_phys_skip"]
                                        (default True). Zero-initialised output layer →
                                        identity at startup, backward-compatible.

z_var improvement (improvement 5):
  5. Variability latent z_var        — masked intraday std + optional ramp_rate of K*
                                        projected to z_var_dim dims and normalised
                                        by a running EMA (mean/std) so z_var lives on
                                        the same scale as z_flat (~N(0,1) at steady state)
                                        concatenated onto z_flat → z_full passed to diffusion
                                        decoder only sees z_flat → no generation mismatch

Regime classification uses 4 classes derived entirely from the fitted RegimeGMM
(clear / mixed-clear / mixed-overcast / overcast). No thresholds are hardcoded
anywhere in this file; all numeric parameters come from config.

Config keys for z_var (all under vae.*):
    z_var_dim           int   number of z_var output dimensions (default 4)
    z_var_proj_hidden   int   hidden size of z_var projection MLP (default 32)
    z_var_ema_decay     float EMA decay for running normalisation (default 0.99)
    z_var_use_ramp      bool  include masked ramp_rate feature in z_var input (default True)

Losses:
    L_recon  = masked L1(K̂*, K*)
    L_fft    = whitened FFT magnitude MSE
    L_phase  = phase_weight · mean |∠FFT(k̂) − ∠FFT(k*)| at significant freqs
    L_ramp   = multi-offset ramp MAE
    L_curv   = curvature_weight · masked MAE on Δ²K*
    L_spec   = spectral_weight · (L_fft + L_phase + L_ramp + L_curv)
    L_sep    = 4-class prototype separation (clear / mx_clear / mx_overcast / overcast)
    L_var    = soft lower-bound on per-dim latent std
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Per-timestep FiLM  (encoder blocks)
# ─────────────────────────────────────────────────────────────────────────────

class _FiLMTimestep(nn.Module):
    """Per-timestep FiLM for encoder conv blocks.

    h' = (1 + γ(p_t)) ⊙ h + β(p_t)

    Physics are pooled from T_sun → T_conv via adaptive average pooling to match
    the downsampled conv feature map.
    """

    def __init__(self, physics_dim: int, out_ch: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h:       torch.Tensor,   # (B, C, T_conv)
        physics: torch.Tensor,   # (B, T_sun, physics_dim)
    ) -> torch.Tensor:
        B, C, T_conv = h.shape
        p = physics.permute(0, 2, 1).float()
        if p.shape[2] != T_conv:
            p = F.adaptive_avg_pool1d(p, T_conv)
        p = p.permute(0, 2, 1)
        gb = self.mlp(p)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.permute(0, 2, 1) + 1.0
        beta  = beta.permute(0, 2, 1)
        return gamma * h + beta


# ─────────────────────────────────────────────────────────────────────────────
# Encoder conv block
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock1d(nn.Module):
    """Conv1d(stride) → GroupNorm(1,C) → [_FiLMTimestep(physics)] → GELU."""

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int,
        kernel:      int,
        stride:      int,
        physics_dim: int = 0,
        film_hidden: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2)
        self.norm = nn.GroupNorm(1, out_ch)
        self.film = (_FiLMTimestep(physics_dim, out_ch, film_hidden)
                     if physics_dim > 0 else None)
        self.act  = nn.GELU()

    def forward(
        self,
        x:       torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(self.conv(x))
        if self.film is not None and physics is not None:
            h = self.film(h, physics)
        return self.act(h)


# ─────────────────────────────────────────────────────────────────────────────
# Global FiLM  (decoder blocks)
# ─────────────────────────────────────────────────────────────────────────────

class _FiLM(nn.Module):
    """Global FiLM for decoder: h' = (1 + γ(p̄)) ⊙ h + β(p̄).

    p̄ = masked mean of intraday physics over valid timesteps.
    Final layer zero-initialised → identity at init.
    """

    def __init__(self, physics_dim: int, out_ch: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h:          torch.Tensor,
        physics:    torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            mf     = valid_mask.unsqueeze(-1).float()
            n      = mf.sum(dim=1).clamp(min=1.0)
            p_mean = (physics * mf).sum(dim=1) / n
        else:
            p_mean = physics.mean(dim=1)
        gamma, beta = self.mlp(p_mean).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1) + 1.0
        beta  = beta.unsqueeze(-1)
        return gamma * h + beta


# ─────────────────────────────────────────────────────────────────────────────
# z-FiLM  (decoder blocks — improvement 1)
# ─────────────────────────────────────────────────────────────────────────────

class _FiLMz(nn.Module):
    """Per-block z injection for the decoder: h' = (1 + γ(z)) ⊙ h + β(z).

    Re-injects the latent code at every decoder depth so the "what kind of day"
    signal is not diluted through 4 deconv stages.  This is particularly
    important for overcast and mixed-overcast days whose z values differ most
    from the clear-sky mode.

    Final layer zero-initialised → identity transform at startup.
    Completely safe at generation time — z is always available.

    Config key:
        vae.decoder_z_film_hidden  int  hidden size of the MLP (default 32)
    """

    def __init__(self, d_z: int, out_ch: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: (B, C, T)   z: (B, d_z)
        gb    = self.mlp(z)                         # (B, C*2)
        gamma, beta = gb.chunk(2, dim=1)            # (B, C) each
        gamma = gamma.unsqueeze(-1) + 1.0           # (B, C, 1)
        beta  = beta.unsqueeze(-1)                  # (B, C, 1)
        return gamma * h + beta


# ─────────────────────────────────────────────────────────────────────────────
# Physics-only skip connection  (improvement 6)
# ─────────────────────────────────────────────────────────────────────────────

class _PhysicsSkip(nn.Module):
    """Per-timestep physics skip connection for decoder blocks.

    Injects a fine-grained physics signal at each decoder resolution level as a
    residual addition to the deconv feature map.

    WHY THIS IS GENERATION-SAFE
    ----------------------------
    The input is ONLY intraday_phys — the deterministic solar geometry tensor
    (zenith_norm, ETR_norm, GCS_norm + optional Δzenith, ΔETR).  This is always
    available at generation time because it is computed from ephemeris alone, with
    no dependence on observed K* or obs_phys channels.

    The encoder is completely uninvolved.  This is NOT a U-Net skip carrying
    encoder feature maps — it is a physics projection skip that provides the
    decoder with per-timestep solar geometry context at each deconv depth.

    PROBLEM SOLVED
    --------------
    The existing FiLM conditioning injects physics as a *global* (masked-mean)
    scale/shift.  After 4× downsampling, the decoder at the coarsest level sees
    physics averaged over the whole day — it cannot tell whether a specific
    timestep is at solar noon vs dawn.  The physics skip restores this per-timestep
    resolution at every decoder depth by projecting the (B, T, dec_physics_dim)
    tensor to (B, out_ch, T) and adding it to the feature map.

    ARCHITECTURE
    ------------
        skip = Linear(dec_physics_dim → out_ch, applied per-timestep)
             + zero-init output → identity at startup, backward-compatible

    The physics tensor is adaptively interpolated to match the current deconv
    output length T_dec (which varies by block depth), so no shape mismatches.

    Config key:
        vae.decoder_phys_skip  bool  enable/disable (default True)
        vae.phys_skip_hidden   int   hidden dim of the projection MLP (default 0
                                     means a single linear; set >0 for nonlinear)
    """

    def __init__(self, dec_physics_dim: int, out_ch: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            # Two-layer MLP for richer physics embedding per timestep
            self.proj = nn.Sequential(
                nn.Linear(dec_physics_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_ch),
            )
            # Zero-init final layer → no effect at startup
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)
        else:
            # Single linear projection — simplest form
            self.proj = nn.Linear(dec_physics_dim, out_ch)
            # Zero-init → identity at startup
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        h:      torch.Tensor,   # (B, out_ch, T_dec)  — deconv output
        phys:   torch.Tensor,   # (B, T_sun, dec_physics_dim)  — augmented physics
    ) -> torch.Tensor:
        """Add per-timestep physics residual to deconv feature map.

        phys is interpolated from T_sun → T_dec so the residual aligns with
        the current decoder resolution regardless of stride/depth.
        """
        B, C, T_dec = h.shape

        # Permute to (B, T_sun, dec_physics_dim) for linear, then back
        p = phys.float()                                    # (B, T_sun, D_phys)
        skip = self.proj(p)                                 # (B, T_sun, out_ch)
        skip = skip.permute(0, 2, 1)                        # (B, out_ch, T_sun)

        # Interpolate physics resolution to match current deconv output length
        if skip.shape[2] != T_dec:
            skip = F.interpolate(skip, size=T_dec, mode="linear", align_corners=False)

        return h + skip                                     # residual addition


# ─────────────────────────────────────────────────────────────────────────────
# Decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class _DeconvBlock1d(nn.Module):
    """ConvTranspose1d(stride) → GroupNorm → global FiLM(physics) → z-FiLM(z)
    → physics skip (improvement 6) → GELU.

    Improvement 1: z-FiLM re-injects the latent code at this depth so the
    "what kind of day" signal is not diluted through the deconv stack.
    Zero-initialised → identity at startup, backward-compatible.

    Improvement 6: _PhysicsSkip adds a per-timestep physics residual after the
    FiLM conditioning.  This gives each block fine-grained solar geometry context
    at its own resolution level, not just the global mean from FiLM.
    Zero-initialised → identity at startup.  Generation-safe (physics only).
    Pass phys_skip=None to disable per-block (controlled by AEDecoder).
    """

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int,
        kernel:      int,
        stride:      int,
        physics_dim: int,
        film_hidden: int,
        d_z:         int = 0,
        z_film_hidden: int = 32,
        # Improvement 6: physics skip — pass dec_physics_dim and hidden to enable
        phys_skip_dim:    int = 0,   # 0 = disabled
        phys_skip_hidden: int = 0,   # 0 = single linear; >0 = MLP with hidden layer
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride,
            padding=kernel // 2, output_padding=stride - 1,
        )
        self.norm    = nn.GroupNorm(1, out_ch)
        self.film    = _FiLM(physics_dim, out_ch, film_hidden)
        self.film_z  = _FiLMz(d_z, out_ch, z_film_hidden) if d_z > 0 else None

        # Improvement 6: physics skip — None when disabled (phys_skip_dim == 0)
        self.phys_skip = (
            _PhysicsSkip(phys_skip_dim, out_ch, hidden=phys_skip_hidden)
            if phys_skip_dim > 0 else None
        )

        self.act = nn.GELU()

    def forward(
        self,
        x:          torch.Tensor,
        physics:    torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        z:          Optional[torch.Tensor] = None,
        phys_aug:   Optional[torch.Tensor] = None,   # (B, T_sun, dec_physics_dim) for skip
    ) -> torch.Tensor:
        h = self.norm(self.deconv(x))
        h = self.film(h, physics, valid_mask=valid_mask)
        if self.film_z is not None and z is not None:
            h = self.film_z(h, z)
        # Improvement 6: add per-timestep physics residual if enabled
        # phys_aug is the full-resolution augmented physics (B, T_sun, dec_physics_dim).
        # _PhysicsSkip handles the interpolation to the current deconv length internally.
        if self.phys_skip is not None and phys_aug is not None:
            h = self.phys_skip(h, phys_aug)
        return self.act(h)


# ─────────────────────────────────────────────────────────────────────────────
# z_var head  (improvement 5)
# ─────────────────────────────────────────────────────────────────────────────

class _ZVarHead(nn.Module):
    """Computes a variability latent z_var from K* statistics.

    Input features computed from (k_star, valid_mask) — no conv features, no
    encoder weights involved.  This means the features are:
        - fully available at training time (k_star is the encoder input)
        - ABSENT at generation time, which is correct: the diffusion model
          generates z_var jointly with z_flat as part of z_full

    Features (2 or 3 scalars per sample, depending on z_var_use_ramp):
        intraday_std   : masked std of K* over valid timesteps
        mean_k         : masked mean of K* (separates clear from overcast level)
        ramp_rate      : mean |ΔK*| over valid transitions  [if z_var_use_ramp]

    These are projected to z_var_dim dimensions via a small MLP.

    Normalisation:
        A running EMA of (feature_mean, feature_std) is maintained during
        training so that z_var lives on approximately the same scale as z_flat
        at steady state.  The EMA is a non-parameter buffer updated only during
        forward() when self.training is True (never during validation/inference).

    Config keys (all under vae.*):
        z_var_dim          int   output dims concatenated onto z_flat (default 4)
        z_var_proj_hidden  int   MLP hidden size (default 32)
        z_var_ema_decay    float EMA decay for running normalisation (default 0.99)
        z_var_use_ramp     bool  include ramp_rate feature (default True)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        vc              = cfg["vae"]
        self.d_out      = int(vc.get("z_var_dim", 4))
        hidden          = int(vc.get("z_var_proj_hidden", 32))
        self.ema_decay  = float(vc.get("z_var_ema_decay", 0.99))
        self.use_ramp   = bool(vc.get("z_var_use_ramp", True))
        self.n_feats    = 3 if self.use_ramp else 2

        self.mlp = nn.Sequential(
            nn.Linear(self.n_feats, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_out),
        )
        # Running EMA normalisation buffers (not parameters — never in state_dict
        # for optimiser; saved in checkpoint via register_buffer so they persist).
        self.register_buffer("_feat_mean", torch.zeros(self.n_feats))
        self.register_buffer("_feat_std",  torch.ones(self.n_feats))
        self._ema_initialised = False

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_features(
        k_star:     torch.Tensor,   # (B, T)
        valid_mask: torch.Tensor,   # (B, T) bool
        use_ramp:   bool,
    ) -> torch.Tensor:              # (B, n_feats)
        mf      = valid_mask.float()
        n_valid = mf.sum(dim=1).clamp(min=1.0)          # (B,)

        k_mean  = (k_star * mf).sum(dim=1) / n_valid    # (B,)
        k_var   = ((k_star - k_mean.unsqueeze(1)) ** 2 * mf).sum(dim=1) / n_valid
        k_std   = k_var.clamp(min=0.0).sqrt()           # (B,)

        feats = [k_std, k_mean]

        if use_ramp:
            # mean |ΔK*| over valid consecutive pairs
            mask_shift = mf[:, :-1] * mf[:, 1:]         # (B, T-1)
            n_pairs    = mask_shift.sum(dim=1).clamp(min=1.0)
            ramp_rate  = ((k_star[:, 1:] - k_star[:, :-1]).abs() * mask_shift
                          ).sum(dim=1) / n_pairs          # (B,)
            feats.append(ramp_rate)

        return torch.stack(feats, dim=1)                 # (B, n_feats)

    # ------------------------------------------------------------------
    def _normalise(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalise features using running EMA stats; update EMA during training.

        FIX H3: Added fast-warmup EMA and convergence diagnostics.

        Root cause of original asymmetry: EMA decay=0.99 is very slow.
        With ~200 batches/epoch, the EMA needs ~5 epochs to approach the true
        feature mean. During this period z_var dims carry a large systematic
        bias (e.g. mean_k≈0.7 for clear-dominated data enters the diffusion
        model un-centred), producing the asymmetric dim distributions observed.

        Fix 1 — Fast warmup: use decay=0.50 for the first 20 updates, then
        ramp up to the configured decay over the next 80 updates.
        At decay=0.50 the EMA reaches 99% of the true mean in ~7 batches.
        At decay=0.99 it takes ~460 batches. Warmup closes this gap.

        Fix 2 — Convergence diagnostic: log _feat_mean and _feat_std every
        500 forward calls so asymmetry is visible without running diagnostics.
        """
        if self.training:
            batch_mean = feats.mean(dim=0).detach()
            batch_std  = feats.std(dim=0).clamp(min=1e-6).detach()
            if not self._ema_initialised:
                # Initialise with first-batch statistics for immediate accuracy.
                self._feat_mean.copy_(batch_mean)
                self._feat_std.copy_(batch_std)
                self._ema_initialised = True
                self._ema_step = 0
            else:
                self._ema_step = getattr(self, "_ema_step", 0) + 1
                # FIX H3-1: Fast warmup — high learning rate for first 20 updates,
                # then blend toward the configured slow decay over the next 80.
                # This converges to the true data mean within the first epoch
                # rather than needing 5+ epochs to reach 99% convergence.
                if self._ema_step < 20:
                    d = 0.50   # fast: reach 99% of true mean in ~7 batches
                elif self._ema_step < 100:
                    # linear ramp from 0.50 to configured decay over 80 steps
                    frac = (self._ema_step - 20) / 80.0
                    d = 0.50 + frac * (self.ema_decay - 0.50)
                else:
                    d = self.ema_decay   # steady-state slow tracking
                self._feat_mean.mul_(d).add_(batch_mean * (1.0 - d))
                self._feat_std.mul_(d).add_(batch_std  * (1.0 - d))
            # FIX H3-2: Log EMA state every 500 updates to surface asymmetry.
            _step = getattr(self, "_ema_step", 0)
            if _step % 500 == 0:
                import logging as _lg
                _lg.getLogger(__name__).debug(
                    "[z_var EMA step=%d] feat_mean=%s  feat_std=%s  "
                    "(decay=%.3f  initialised=%s)",
                    _step,
                    [f"{v:.3f}" for v in self._feat_mean.tolist()],
                    [f"{v:.3f}" for v in self._feat_std.tolist()],
                    self.ema_decay,
                    self._ema_initialised,
                )

        return (feats - self._feat_mean) / self._feat_std.clamp(min=1e-6)

    # ------------------------------------------------------------------
    def forward(
        self,
        k_star:     torch.Tensor,   # (B, T)
        valid_mask: torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:              # (B, d_out)
        feats = self._compute_features(k_star, valid_mask, self.use_ramp)
        feats = self._normalise(feats)
        return self.mlp(feats)

    def force_reset_ema(self) -> None:
        """FIX H3: Reset EMA state so a resumed run starts fresh.

        Called by train_vae() after loading a checkpoint. Without this,
        stale EMA state from a previous run (potentially with different data
        distribution or collapsed z_std) persists into the new run and causes
        the first epoch to have systematically wrong z_var normalisation.
        """
        self._feat_mean.zero_()
        self._feat_std.fill_(1.0)
        self._ema_initialised = False
        self._ema_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class AEEncoder(nn.Module):
    """Deterministic 1-D convolutional encoder.

    Input : (B, T_padded, 2 + physics_dim + N_obs)
            channels = [K*, ΔK*, zenith_norm, ETR_norm, GCS_norm, obs_channels...]
    Output: z (B, d_z), [] — empty list for API compatibility.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        vc          = cfg["vae"]
        n_obs       = len(vc.get("obs_channels", []))
        physics_dim = vc.get("intraday_physics_dim", 3)
        film_hidden = vc.get("encoder_film_hidden", 32)
        in_ch       = 2 + physics_dim + n_obs
        channels    = vc["encoder_channels"]
        kernel      = vc["kernel_size"]
        stride      = vc["stride"]
        d_z         = vc["latent_dim"]

        ch_in  = in_ch
        layers = []
        for ch_out in channels:
            layers.append(
                _ConvBlock1d(ch_in, ch_out, kernel, stride,
                             physics_dim=physics_dim, film_hidden=film_hidden)
            )
            ch_in = ch_out
        self.convs = nn.ModuleList(layers)

        head_dim         = max(channels[-1] // 8, 8)
        self.attn_q      = nn.Linear(channels[-1], head_dim, bias=False)
        self.attn_k      = nn.Linear(channels[-1], head_dim, bias=False)
        self._attn_scale = head_dim ** -0.5

        self.dropout = nn.Dropout(vc.get("encoder_dropout", 0.2))
        self.z_head  = nn.Linear(channels[-1], d_z)

    def attn_pool(
        self,
        h:          torch.Tensor,   # (B, C, T_conv)
        valid_mask: torch.Tensor,   # (B, T_sun) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Key-query attention pool → (B, C) pooled, (B, T_conv) weights."""
        B, C, T_conv = h.shape
        mask_down = F.adaptive_max_pool1d(
            valid_mask.float().unsqueeze(1), T_conv
        ).squeeze(1)

        h_t = h.permute(0, 2, 1)
        mf     = mask_down.unsqueeze(-1)
        n_v    = mf.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_mean = (h_t * mf).sum(dim=1, keepdim=True) / n_v
        q = self.attn_q(h_mean)
        k = self.attn_k(h_t)

        scores = (q * k).sum(dim=-1) * self._attn_scale
        scores = scores.masked_fill(mask_down < 0.5, float("-inf"))

        all_masked = ~torch.isfinite(scores).any(dim=1)
        if all_masked.any():
            scores = scores.clone()
            scores[all_masked] = 0.0

        weights = torch.softmax(scores, dim=1)
        pooled  = (h_t * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(
        self,
        x:          torch.Tensor,
        valid_mask: torch.Tensor,
        physics:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        expected = self.convs[0].conv.in_channels
        assert x.shape[-1] == expected, (
            f"AEEncoder channel mismatch: got {x.shape[-1]}, expected {expected}."
        )
        x = x * valid_mask.unsqueeze(-1).float()
        h = x.permute(0, 2, 1)
        for blk in self.convs:
            h = blk(h, physics=physics)
        h, _ = self.attn_pool(h, valid_mask)
        h = self.dropout(h)
        return self.z_head(h), []


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class AEDecoder(nn.Module):
    """Physics-conditioned 1-D convolutional decoder with physics-only skip connections.

    Input : z (B, d_z) + physics (B, T_sun, physics_dim) + target_len int
    Output: K̂* (B, T_sun) in (0, k_max]

    Improvements vs original (all generation-safe — no encoder signals used):

    1. z-FiLM at every deconv block
       Each _DeconvBlock1d now receives z and applies γ(z),β(z) after the
       physics FiLM.  Zero-init → identity at startup.

    2. Δphysics input channels (Δzenith, ΔETR)
       Two extra channels appended to the physics tensor fed into the decoder:
       the temporal differences of zenith_norm and ETR_norm. These encode the
       solar geometry ramp rate — available at generation time without K*.
       Controlled by cfg["vae"]["decoder_delta_physics"] (default True).

    3. Depthwise-separable output refinement
       Replaces the single Conv1d(C, C, k=3) with:
         depthwise  Conv1d(C, C, k=3, groups=C)  — local intraday shape
         pointwise  Conv1d(C, C, k=1)             — channel mixing
       Same parameter count, better capacity for shape prediction.

    4. Physics bypass lane
       A small branch computes k_bypass = sigmoid(Linear(physics_dim, 1))·k_max
       from the masked physics mean. A learned scalar α (init=0) blends it into
       the output: k_out = k_decoder + α·k_bypass.
       α=0 at startup → no effect. Gradient flows freely so α grows only if the
       bypass helps. Gives overcast days an escape hatch when z is uninformative.

    6. Physics-only skip connections (improvement 6)
       At each deconv block, a _PhysicsSkip module projects the full-resolution
       intraday_phys (B, T_sun, dec_physics_dim) to the block's channel width and
       adds it as a per-timestep residual.  This gives every decoder block access
       to precise solar geometry at its own resolution level — not just the global
       mean that FiLM provides.
       GENERATION-SAFE: only uses intraday_phys (deterministic ephemeris).
       Zero-init → identity at startup.
       Controlled by cfg["vae"]["decoder_phys_skip"] (default True).
       MLP hidden width: cfg["vae"]["phys_skip_hidden"] (default 0 = linear only).

    Config keys:
        vae.decoder_z_film_hidden   int   hidden for z-FiLM MLPs (default 32)
        vae.decoder_delta_physics   bool  add Δzenith,ΔETR to decoder (default True)
        vae.decoder_z_proj_dim      int   z broadcast projection dim (default 32)
        vae.decoder_phys_skip       bool  enable physics skip connections (default True)
        vae.phys_skip_hidden        int   hidden dim in skip MLP; 0 = linear (default 0)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        vc           = cfg["vae"]
        d_z          = vc["latent_dim"]
        channels     = list(reversed(vc["encoder_channels"]))
        kernel       = vc["kernel_size"]
        stride       = vc["stride"]
        film_hidden  = vc["film_hidden_dim"]
        physics_dim  = vc.get("intraday_physics_dim", 3)
        self.k_max   = cfg["physics"]["k_max"]
        self.d_z     = d_z

        # Improvement 2: Δphysics — optionally extend physics_dim by 2
        self._use_delta_physics = bool(vc.get("decoder_delta_physics", True))
        self._dec_physics_dim   = physics_dim + (2 if self._use_delta_physics else 0)

        init_len      = vc.get("decoder_init_len", 4)
        self.proj     = nn.Linear(d_z, channels[0] * init_len)
        self.init_ch  = channels[0]
        self.init_len = init_len

        z_film_hidden = int(vc.get("decoder_z_film_hidden", 32))

        # Improvement 6: physics skip connection config
        # decoder_phys_skip=True injects per-timestep physics at every deconv block.
        # phys_skip_hidden=0 means a single linear projection (cheapest option).
        # Set phys_skip_hidden=16 or 32 for a small nonlinear MLP if needed.
        self._use_phys_skip   = bool(vc.get("decoder_phys_skip", True))
        phys_skip_hidden      = int(vc.get("phys_skip_hidden", 0))
        # phys_skip_dim is dec_physics_dim when skips are enabled, else 0 (disabled)
        phys_skip_dim = self._dec_physics_dim if self._use_phys_skip else 0

        # Improvement 1+6: pass d_z, z_film_hidden, and phys_skip params to every block
        pairs = list(zip(channels[:-1], channels[1:])) + [(channels[-1], channels[-1])]
        self.blocks = nn.ModuleList([
            _DeconvBlock1d(
                ci, co, kernel, stride,
                self._dec_physics_dim, film_hidden,
                d_z=d_z, z_film_hidden=z_film_hidden,
                phys_skip_dim=phys_skip_dim,       # improvement 6: >0 enables skip
                phys_skip_hidden=phys_skip_hidden,  # improvement 6: 0 = linear
            )
            for ci, co in pairs
        ])

        z_proj_dim = vc.get("decoder_z_proj_dim", 32)
        self.z_proj = nn.Linear(d_z, z_proj_dim)
        out_in = channels[-1] + self._dec_physics_dim + z_proj_dim

        # Improvement 3: depthwise-separable output refinement
        self.out_refine_dw = nn.Conv1d(out_in, out_in, kernel_size=3,
                                       padding=1, groups=out_in)   # depthwise
        self.out_refine_pw = nn.Conv1d(out_in, out_in, kernel_size=1)  # pointwise
        self.out_act       = nn.GELU()
        self.out_conv      = nn.Conv1d(out_in, 1, kernel_size=1)

        # Improvement 4: physics bypass lane
        # k_bypass = sigmoid(Linear(dec_physics_dim, 1)) * k_max
        # k_out = k_decoder + alpha * k_bypass
        self.bypass_proj  = nn.Linear(self._dec_physics_dim, 1)
        # bypass_alpha controls how much the physics bypass lane contributes.
        # Init at -3.0 → sigmoid(-3.0) = 0.047 (≈5% contribution at startup).
        # This forces k_dec to learn the full profile shape before bypass grows.
        # bypass_alpha is a free parameter — it will only grow during training
        # if it genuinely reduces loss. k_dec and bypass are jointly optimised
        # so whatever bypass_alpha converges to is part of the learned decoder.
        #
        # DO NOT zero bypass_alpha at inference (old FIX A was wrong):
        # k_dec is trained jointly with bypass; removing bypass at inference
        # creates a decoder mismatch — k_dec was never trained to work alone.
        # The -3.0 init is the correct and sufficient fix for the flat-floor
        # artifact seen with the original 0.0 init (sigmoid(0)=0.5 floor).
        #
        # [next train] Verify bypass_alpha converged below ~-1.5 (sigmoid<0.18).
        # If it grows above 0.0 (sigmoid>0.5), the bypass is dominating again —
        # raise the reconstruction loss weight or add an explicit bypass penalty.
        # Monitor via the inference log: "bypass_alpha at inference: sigmoid=X".
        self.bypass_alpha = nn.Parameter(torch.full((1,), -3.0))  # sigmoid(-3)=0.047
        nn.init.zeros_(self.bypass_proj.weight)
        nn.init.zeros_(self.bypass_proj.bias)

    def _make_dec_physics(
        self,
        physics:    torch.Tensor,    # (B, T_sun, physics_dim)  original
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Optionally append Δzenith, ΔETR to physics tensor (improvement 2).

        Δp_t = p_t − p_{t-1}, masked so only differences between two valid
        timesteps are non-zero.  First valid timestep gets Δ=0.
        This mirrors the ΔK* channel in the encoder, giving the decoder a ramp
        signal derived purely from solar geometry — available at generation time.
        """
        if not self._use_delta_physics:
            return physics

        # Compute differences for the first two physics channels (zenith, ETR).
        # GCS (channel 2) is already a smooth scaled irradiance — no ramp needed.
        p2 = physics[..., :2]                             # (B, T, 2)
        dp = torch.zeros_like(p2)
        dp[:, 1:] = p2[:, 1:] - p2[:, :-1]
        if valid_mask is not None:
            dp[:, 1:] = dp[:, 1:] * valid_mask[:, :-1].unsqueeze(-1).float()

        return torch.cat([physics, dp], dim=-1)           # (B, T, physics_dim+2)

    def forward(
        self,
        z:           torch.Tensor,
        physics:     torch.Tensor,
        target_len:  int,
        valid_mask:  Optional[torch.Tensor] = None,
        skips:       Optional[list] = None,   # unused — kept for API compat
        use_encoder: bool = False,            # unused — kept for API compat
    ) -> torch.Tensor:
        # Build augmented physics (+ Δphysics if enabled)
        phys_aug = self._make_dec_physics(physics, valid_mask)  # (B,T,dec_physics_dim)

        # Seed feature map from z
        h = self.proj(z).view(z.size(0), self.init_ch, self.init_len)

        # Improvement 1+2+6: deconv blocks with z-FiLM, augmented physics, and
        # physics skip connections.  phys_aug is passed to each block so the
        # _PhysicsSkip module inside can interpolate it to the block's resolution.
        # When decoder_phys_skip=False, phys_skip is None and phys_aug is ignored.
        for blk in self.blocks:
            h = blk(h, phys_aug, valid_mask=valid_mask, z=z, phys_aug=phys_aug)

        h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)

        # Per-timestep physics (augmented) broadcast to target_len
        phys_t = phys_aug.permute(0, 2, 1).float()
        if phys_t.shape[2] != target_len:
            phys_t = F.interpolate(phys_t, size=target_len, mode="linear",
                                   align_corners=False)

        # z broadcast over time
        z_t = self.z_proj(z).unsqueeze(-1).expand(-1, -1, target_len)
        h   = torch.cat([h, phys_t, z_t], dim=1)

        # Improvement 3: depthwise-separable refinement
        h = self.out_act(self.out_refine_pw(self.out_refine_dw(h)))
        k_dec = torch.sigmoid(self.out_conv(h).squeeze(1)) * self.k_max

        # Improvement 4: physics bypass lane
        # Compute physics mean (masked) → scalar GCS-based baseline
        if valid_mask is not None:
            mf      = valid_mask.unsqueeze(-1).float()
            n       = mf.sum(dim=1).clamp(min=1.0)
            p_mean  = (phys_aug * mf).sum(dim=1) / n       # (B, dec_physics_dim)
        else:
            p_mean  = phys_aug.mean(dim=1)
        k_bypass = torch.sigmoid(self.bypass_proj(p_mean)).squeeze(-1) * self.k_max
        # bypass_alpha: learned scalar in (-∞, +∞), applied as sigmoid → (0,1).
        # Initialised at -3.0 so it starts near-zero and only grows if helpful.
        # Jointly optimised with k_dec — do not override at inference.
        # [next train] Log "bypass_alpha at inference: sigmoid=X" after loading
        # checkpoint. Target: sigmoid(bypass_alpha) < 0.18. If higher, the bypass
        # is dominating and the -3.0 init did not take effect (old checkpoint).
        alpha = torch.sigmoid(self.bypass_alpha)         # scalar ∈ (0,1)
        # FIX NEW-A: clamp final output to (eps, k_max].
        # k_dec ∈ (0, k_max] and alpha*k_bypass ∈ (0, alpha*k_max], so their
        # sum can reach up to 2*k_max without clamping. Observed khat_max=1.397
        # with k_max=1.30. This saturates the sigmoid gradient and causes loss
        # spikes. Clamp here (not inside k_dec) so bypass lane still has gradient.
        k_out = k_dec + alpha * k_bypass.unsqueeze(-1).expand_as(k_dec)
        return k_out.clamp(1e-6, self.k_max)


# ─────────────────────────────────────────────────────────────────────────────
# Full Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class SolarVAE(nn.Module):
    """Physics-conditioned deterministic autoencoder for daily K* profiles.

    encode() → (z_full, [])
        z_full = cat(z_flat, z_var)  shape (B, d_z + z_var_dim)
        z_flat  — shape latent, fed to decoder via z-FiLM
        z_var   — variability latent, fed to diffusion model only
        Pass z_full to regime_separation_loss() and the diffusion model.

    decode(z_full_or_flat, ...) → K̂* (B, T_sun) ∈ (0, k_max]
        Accepts either z_full (B, d_z + z_var_dim) OR z_flat (B, d_z).
        Always slices z_full[:, :d_z] so the decoder only ever sees z_flat.
        At generation time the diffusion model produces z_full; passing it
        here is safe — the z_var tail is silently ignored by the decoder.

    forward() → (K̂*, z_full)

    GENERATION-TIME CONTRACT:
        Diffusion model generates z_full (d_z + z_var_dim dims) jointly.
        Decoder receives z_full[:, :d_z] = z_flat.  Encoder absent.  No mismatch.

    Config key added:
        vae.z_var_dim  int  — number of z_var dimensions (default 4).
        Also see _ZVarHead for additional config keys.
        The diffusion model's input_dim must be set to d_z + z_var_dim = d_z_full.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.encoder  = AEEncoder(cfg)
        self.decoder  = AEDecoder(cfg)
        self.z_var_head = _ZVarHead(cfg)
        self.d_z      = cfg["vae"]["latent_dim"]           # z_flat dims
        self.d_z_full = self.d_z + self.z_var_head.d_out  # z_flat + z_var dims
        self.k_max    = cfg["physics"]["k_max"]

    @staticmethod
    def _delta_k(
        k_star:     torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ΔK*_t = K*_t − K*_{t-1}, masked so only differences between two valid
        timesteps are non-zero. First valid timestep gets ΔK*=0."""
        d = torch.zeros_like(k_star)
        d[:, 1:] = (k_star[:, 1:] - k_star[:, :-1]) * valid_mask[:, :-1].float()
        return d

    def encode(
        self,
        k_star:        torch.Tensor,
        intraday_phys: torch.Tensor,
        valid_mask:    torch.Tensor,
        obs_phys:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Returns (z_full, []) where z_full = cat(z_flat, z_var).

        z_full shape: (B, d_z + z_var_dim)  — pass this to the diffusion model
                                               and to regime_separation_loss().
        The decoder slices z_full[:, :d_z] internally, so z_full is always safe
        to pass to decode() regardless of whether z_var dims are appended.
        """
        dk    = self._delta_k(k_star, valid_mask)
        parts = [k_star.unsqueeze(-1), dk.unsqueeze(-1), intraday_phys]
        if obs_phys is not None and obs_phys.shape[-1] > 0:
            parts.append(obs_phys)
        x      = torch.cat(parts, dim=-1)
        z_flat, skips = self.encoder(x, valid_mask, physics=intraday_phys)
        z_var  = self.z_var_head(k_star, valid_mask)          # (B, z_var_dim)
        z_full = torch.cat([z_flat, z_var], dim=1)            # (B, d_z + z_var_dim)
        return z_full, skips

    def encode_with_attn_entropy(
        self,
        k_star:        torch.Tensor,
        intraday_phys: torch.Tensor,
        valid_mask:    torch.Tensor,
        obs_phys:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(z_full, attn_entropy) — entropy diagnostic for training health logging.

        z_full = cat(z_flat, z_var) as in encode().
        attn_entropy is computed from the attention weights of the z_flat encoder
        only (z_var is a direct statistic of K*, not attention-derived).
        """
        dk    = self._delta_k(k_star, valid_mask)
        parts = [k_star.unsqueeze(-1), dk.unsqueeze(-1), intraday_phys]
        if obs_phys is not None and obs_phys.shape[-1] > 0:
            parts.append(obs_phys)
        x = torch.cat(parts, dim=-1) * valid_mask.unsqueeze(-1).float()
        h = x.permute(0, 2, 1)
        for blk in self.encoder.convs:
            h = blk(h, physics=intraday_phys)
        pooled, weights = self.encoder.attn_pool(h, valid_mask)
        h      = self.encoder.dropout(pooled)
        z_flat = self.encoder.z_head(h)
        z_var  = self.z_var_head(k_star, valid_mask)
        z_full = torch.cat([z_flat, z_var], dim=1)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
        return z_full, entropy

    def decode(
        self,
        z:             torch.Tensor,   # (B, d_z) OR (B, d_z + z_var_dim) — both accepted
        intraday_phys: torch.Tensor,
        target_len:    int,
        valid_mask:    Optional[torch.Tensor] = None,
        skips:         Optional[list] = None,
        use_encoder:   bool = False,
    ) -> torch.Tensor:
        """Decode z to K̂*.

        Accepts either z_flat (B, d_z) or z_full (B, d_z + z_var_dim).
        Always slices z[:, :d_z] so only z_flat reaches the decoder.
        This means you can pass z_full from encode() or from the diffusion
        model at generation time without any changes — z_var is silently dropped.
        """
        z_flat = z[:, :self.d_z]   # safe for both z_flat and z_full inputs
        return self.decoder(z_flat, intraday_phys, target_len, valid_mask=valid_mask)

    def forward(
        self,
        k_star:        torch.Tensor,
        intraday_phys: torch.Tensor,
        valid_mask:    torch.Tensor,
        obs_phys:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (k_hat, z_full).

        z_full = cat(z_flat, z_var) shape (B, d_z + z_var_dim).
        Pass z_full to regime_separation_loss() — the loss uses the full code
        so separation is enforced across the complete latent representation.
        The decoder internally slices z_flat = z_full[:, :d_z].
        """
        z_full, _ = self.encode(k_star, intraday_phys, valid_mask, obs_phys)
        k_hat     = self.decode(z_full, intraday_phys, k_star.shape[-1],
                                valid_mask=valid_mask)
        return k_hat, z_full


# ─────────────────────────────────────────────────────────────────────────────
# Loss: ae_loss
# ─────────────────────────────────────────────────────────────────────────────

def ae_loss(
    k_hat:               torch.Tensor,
    k_star:              torch.Tensor,
    valid_mask:          torch.Tensor,
    spectral_w:          float,
    ramp_threshold:      float,
    sample_weight:       Optional[torch.Tensor] = None,
    spectral_power_mean: Optional[torch.Tensor] = None,
    phase_weight:        float = 0.0,
    curvature_weight:    float = 0.0,
    phase_amp_threshold: float = 0.0,
    ramp_offset_steps:   int   = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked L1 + whitened FFT magnitude + phase + ramp MAE + curvature.

    All numeric parameters are passed in from config — no defaults encode
    design choices. sample_weight is applied to the full per-sample loss
    (L_recon + spectral) so regime-specific loss weighting also applies to
    the spectral terms where ramp fidelity is measured.

    Returns (total_loss, recon_loss).
    """
    mask_f      = valid_mask.float()
    n_valid_per = mask_f.sum(dim=1).clamp(min=1.0)
    B_sz        = k_hat.shape[0]

    # L_recon
    l1_per = (k_hat - k_star).abs().mul(mask_f).sum(dim=1) / n_valid_per
    recon  = (l1_per * sample_weight).mean() if sample_weight is not None \
             else l1_per.mean()

    # L_fft + L_phase
    n_valid_int = valid_mask.sum(dim=1).long()
    n_fft = int(n_valid_int.max().item())
    if spectral_power_mean is not None:
        n_fft = (spectral_power_mean.shape[0] - 1) * 2

    fft_mag_list:   List[torch.Tensor] = []
    fft_phase_list: List[torch.Tensor] = []

    for i in range(B_sz):
        L = int(n_valid_int[i].item())
        if L < 2:
            continue
        fft_h = torch.fft.rfft(k_hat[i,  :L], n=n_fft, norm="ortho")
        fft_r = torch.fft.rfft(k_star[i, :L], n=n_fft, norm="ortho")
        fft_mag_list.append((fft_h.abs() - fft_r.abs()).pow(2))
        amp_mask = (fft_r.abs() > phase_amp_threshold).float()
        fft_phase_list.append((fft_h * fft_r.conj()).angle().abs() * amp_mask)

    fft_per:   List[Optional[torch.Tensor]] = [None] * B_sz
    phase_per: List[Optional[torch.Tensor]] = [None] * B_sz

    if fft_mag_list:
        valid_indices = [i for i in range(B_sz) if int(n_valid_int[i].item()) >= 2]
        assert len(valid_indices) == len(fft_mag_list)
        for list_idx, batch_idx in enumerate(valid_indices):
            fm = fft_mag_list[list_idx]
            ph = fft_phase_list[list_idx]
            if spectral_power_mean is not None:
                w = 1.0 / spectral_power_mean.clamp(min=1e-6)
                w = w / w.mean()
                fft_per[batch_idx]   = (fm * w).mean()
                phase_per[batch_idx] = ph.mean()
            else:
                fft_per[batch_idx]   = fm.mean()
                phase_per[batch_idx] = ph.mean()

    zero = k_hat.new_zeros(1).squeeze()
    fft_per_t   = torch.stack([v if v is not None else zero for v in fft_per])
    phase_per_t = torch.stack([v if v is not None else zero for v in phase_per])

    # L_ramp per sample
    mask_shift = mask_f[:, :-1] * mask_f[:, 1:]
    grad_real  = (k_star[:, 1:] - k_star[:, :-1]) * mask_shift
    grad_hat   = (k_hat[:, 1:]  - k_hat[:, :-1])  * mask_shift
    T1         = grad_real.shape[1]

    ramp_offsets: List[torch.Tensor] = []
    for d in range(-ramp_offset_steps, ramp_offset_steps + 1):
        rs = max(0, -d); re = T1 + min(0, -d)
        hs = max(0,  d); he = T1 + min(0,  d)
        if re <= rs or he <= hs:
            continue
        gr  = grad_real[:, rs:re]
        gh  = grad_hat [:, hs:he]
        rm  = (gr.abs() > ramp_threshold).float()
        n_r = rm.sum(dim=1).clamp(min=1.0)
        ramp_offsets.append(((gr - gh).abs() * rm).sum(dim=1) / n_r)

    ramp_per = (torch.stack(ramp_offsets, dim=0).min(dim=0).values
                if ramp_offsets else k_hat.new_zeros(B_sz))

    # L_curv per sample
    if k_star.shape[1] >= 3:
        c3     = mask_f[:, :-2] * mask_f[:, 1:-1] * mask_f[:, 2:]
        curv_r = k_star[:, 2:] - 2.0 * k_star[:, 1:-1] + k_star[:, :-2]
        curv_h = k_hat[:, 2:]  - 2.0 * k_hat[:, 1:-1]  + k_hat[:, :-2]
        n_c    = c3.sum(dim=1).clamp(min=1.0)
        curv_per = ((curv_h - curv_r).abs() * c3).sum(dim=1) / n_c
    else:
        curv_per = k_hat.new_zeros(B_sz)

    # Build a boolean mask for samples that actually contributed to fft/phase.
    # Short/polar-night days (n_valid < 2) contribute zero, and dividing by B_sz
    # dilutes the spectral gradient in high-latitude batches.
    # Average fft and phase ONLY over valid samples; zero-contribution samples
    # drop out of the spectral mean so their weight falls on ramp+curv instead.
    spectral_valid_mask = (n_valid_int >= 2).float()          # (B_sz,)
    n_spectral_valid    = spectral_valid_mask.sum().clamp(min=1.0)

    # Replace the stacked zero-padded tensors with a masked mean.
    fft_mean   = (fft_per_t   * spectral_valid_mask).sum() / n_spectral_valid
    phase_mean = (phase_per_t * spectral_valid_mask).sum() / n_spectral_valid

    # Rebuild per-sample spectral term with corrected fft/phase:
    # for invalid samples set their fft/phase contribution to the batch mean so
    # the per-sample total is still well-defined for sample_weight weighting.
    fft_per_t_c   = fft_per_t   + (1.0 - spectral_valid_mask) * fft_mean.detach()
    phase_per_t_c = phase_per_t + (1.0 - spectral_valid_mask) * phase_mean.detach()

    spectral_per = (fft_per_t_c
                    + phase_weight     * phase_per_t_c
                    + ramp_per
                    + curvature_weight * curv_per)

    if sample_weight is not None:
        total = (l1_per + spectral_w * spectral_per) * sample_weight
        return total.mean(), recon
    else:
        return (l1_per + spectral_w * spectral_per).mean(), recon


# ─────────────────────────────────────────────────────────────────────────────
# Loss: regime_separation_loss  (4-class)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level EMA registry, keyed by (stage, device, alpha) — train calls only.
# Val calls must NOT update this dict; the caller (_run_vae_epoch) is responsible
# for passing train=True/False and skipping the call during validation.
_sep_ema_registry: Dict[str, float] = {}


def regime_separation_loss(
    z:          torch.Tensor,
    cfg:        Dict,
    regime_ids: torch.Tensor,          # (B,) int64 {0,1,2,3}
    tau0:       Optional[torch.Tensor] = None,
    stage:      str = "vae",
) -> Tuple[torch.Tensor, bool]:
    """Four-class prototype separation in z-space (Stage 1) or τ-space (Stage 2).

    Classes (from RegimeGMM, 4 components):
        0 = clear
        1 = mixed-clear  (partially clear, low variability)
        2 = mixed-overcast (partially cloudy, high variability)
        3 = overcast

    The margin is derived from an EMA of the observed clear–overcast centroid
    distance (d_co), scaled by dynamic_margin_alpha so it sits below the running
    average. The loss fires whenever the current batch d_co falls below the
    historical target, preventing separation from shrinking.

    This function must only be called during training steps. The EMA is global
    state — calling it during validation would corrupt the training margin target.

    Config keys read from cfg["vae"]:
        regime_sep_weight         float  (required)
        dynamic_margin_alpha      float  fraction of EMA to use as target margin
        mixed_margin_scale        float  inner-class margin = margin_co × this
        mixed_sep_weight          float  weight for inner-class separation terms
        regime_sep_min_samples    int    min samples per class to compute loss
        sep_ema_decay             float  EMA decay for margin tracking
    """
    vc       = cfg["vae"]
    sep_w    = float(vc["regime_sep_weight"])
    alpha    = float(vc["dynamic_margin_alpha"])
    mx_scale = float(vc["mixed_margin_scale"])
    mixed_w  = float(vc.get("mixed_sep_weight", sep_w))
    min_samp = int(vc["regime_sep_min_samples"])
    ema_dec  = float(vc.get("sep_ema_decay", 0.98))

    # n_regimes is read from cfg so the overcast index is always n_regimes-1,
    # regardless of whether the GMM has 3, 4, or 5 components.
    # Hardcoding 3 here was a latent bug: safe now (n_components=4) but would
    # silently mis-identify the overcast anchor if n_components ever changed.
    n_regimes   = int(cfg["diffusion"].get("n_regimes", 4))
    oc_idx      = n_regimes - 1   # guaranteed by RegimeGMM: lowest mean_k → last label

    if sep_w == 0.0:
        return z.new_zeros(1).squeeze(), False

    codes = tau0 if tau0 is not None else z

    is_clear    = (regime_ids == 0)
    is_mx_clear = (regime_ids == 1)
    is_mx_oc    = (regime_ids == 2)
    is_overcast = (regime_ids == oc_idx)

    # Need at least clear and overcast to compute anchor distance
    if int(is_clear.sum()) < min_samp or int(is_overcast.sum()) < min_samp:
        return z.new_zeros(1).squeeze(), False

    mu_clear    = codes[is_clear].mean(dim=0)
    mu_overcast = codes[is_overcast].mean(dim=0)

    d_co     = (mu_clear - mu_overcast).norm(p=2)
    d_co_val = d_co.detach().item()

    # EMA margin: tracks a target separation derived from the running average of
    # d_co, scaled by alpha so the loss always provides a small upward push.
    # Seeded at alpha*d_co on first call so the loss fires immediately.
    # Normalise device to base type only (cpu / cuda) so multi-GPU runs
    # and device transitions (e.g. val on CPU) share the same EMA entry.
    _dev_str = str(codes.device).split(":")[0]
    ema_key = f"{stage}_{_dev_str}_{alpha:.4f}"
    if ema_key not in _sep_ema_registry or _sep_ema_registry[ema_key] <= 0.0:
        _sep_ema_registry[ema_key] = max(alpha * d_co_val, 1e-4)
    else:
        _sep_ema_registry[ema_key] = (
            ema_dec * _sep_ema_registry[ema_key] + (1.0 - ema_dec) * d_co_val
        )

    margin_co   = alpha * _sep_ema_registry[ema_key]
    margin_co_t = codes.new_tensor(margin_co)
    margin_mix  = margin_co * mx_scale

    loss  = sep_w * F.relu(margin_co_t - d_co).pow(2)
    fired = bool((d_co < margin_co_t).item())

    # Mixed-clear: should sit between clear and overcast, repelled from both
    if int(is_mx_clear.sum()) >= min_samp:
        mu_mc = codes[is_mx_clear].mean(dim=0)
        d_c_mc  = (mu_clear    - mu_mc).norm(p=2)
        d_oc_mc = (mu_overcast - mu_mc).norm(p=2)
        loss = (loss
                + mixed_w * F.relu(codes.new_tensor(margin_mix) - d_c_mc).pow(2)
                + mixed_w * F.relu(codes.new_tensor(margin_mix) - d_oc_mc).pow(2))
        if not fired:
            fired = bool(
                (d_c_mc < margin_mix).item() or (d_oc_mc < margin_mix).item()
            )

    # Mixed-overcast: should sit between clear and overcast, repelled from both
    if int(is_mx_oc.sum()) >= min_samp:
        mu_mo = codes[is_mx_oc].mean(dim=0)
        d_c_mo  = (mu_clear    - mu_mo).norm(p=2)
        d_oc_mo = (mu_overcast - mu_mo).norm(p=2)
        loss = (loss
                + mixed_w * F.relu(codes.new_tensor(margin_mix) - d_c_mo).pow(2)
                + mixed_w * F.relu(codes.new_tensor(margin_mix) - d_oc_mo).pow(2))
        if not fired:
            fired = bool(
                (d_c_mo < margin_mix).item() or (d_oc_mo < margin_mix).item()
            )

    # Adjacent inner-class separation: mx-clear should be closer to clear than
    # to mx-overcast, and mx-overcast closer to overcast than to mx-clear.
    if int(is_mx_clear.sum()) >= min_samp and int(is_mx_oc.sum()) >= min_samp:
        mu_mc = codes[is_mx_clear].mean(dim=0)
        mu_mo = codes[is_mx_oc].mean(dim=0)
        d_inner = (mu_mc - mu_mo).norm(p=2)
        loss = loss + mixed_w * F.relu(codes.new_tensor(margin_mix) - d_inner).pow(2)
        if not fired:
            fired = bool((d_inner < margin_mix).item())

    return loss, fired


def reset_sep_ema(stage: str = "vae") -> None:
    """Remove all EMA entries for a given stage. Call when resuming from checkpoint
    to prevent stale EMA state from a previous run corrupting the new margin target."""
    keys_to_remove = [k for k in _sep_ema_registry if k.startswith(stage)]
    for k in keys_to_remove:
        del _sep_ema_registry[k]


# ─────────────────────────────────────────────────────────────────────────────
# Loss: latent variance penalty
# ─────────────────────────────────────────────────────────────────────────────

def latent_variance_penalty(
    z:                  torch.Tensor,
    z_std_target:       float,
    variance_weight:    float,
    smoothed_z_std:     Optional[torch.Tensor] = None,
    z_std_upper_target: Optional[float]        = None,
    variance_weight_upper: float               = 0.0,
) -> torch.Tensor:
    """Two-sided soft penalty on per-dimension latent standard deviation.

    Lower bound (always active):
        Fires whenever mean per-dim std < z_std_target.
        Prevents latent collapse after the separation loss is reduced.

    Upper bound (active when z_std_upper_target is provided):
        Fires whenever mean per-dim std > z_std_upper_target.
        Prevents z_std runaway when separation loss is strong — the root cause
        of the gradient explosion seen in training (z_std 0.35→2.4 unchecked).
        Uses a much smaller weight (variance_weight_upper) so it provides
        counter-pressure without dominating the reconstruction loss.

    smoothed_z_std: EMA of per-dim std accumulated across the epoch — prevents
    a single high-variance batch from silencing the lower-bound for a step when
    the global distribution is actually collapsed.

    All parameters are passed from config — no defaults encode design choices.

    Config keys read:
        vae.z_std_target            float  lower-bound target (required)
        vae.z_var_weight            float  weight for lower-bound (required)
        vae.z_std_upper_target      float  upper-bound target (optional; None = off)
        vae.z_var_weight_upper      float  weight for upper-bound (optional; 0 = off)
    """
    if smoothed_z_std is not None:
        current_std = smoothed_z_std.mean()
    else:
        current_std = z.std(dim=0).mean()

    # Lower bound: penalise collapse below target
    loss = variance_weight * F.relu(z_std_target - current_std).pow(2)

    # Upper bound: penalise runaway above upper target (when configured)
    if z_std_upper_target is not None and variance_weight_upper > 0.0:
        loss = loss + variance_weight_upper * F.relu(current_std - z_std_upper_target).pow(2)

    return loss