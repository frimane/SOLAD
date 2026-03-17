
# Stage 2: Transformer denoiser for the τ-space latent diffusion model.

# Each of W day-tokens is formed by summing:
#   - Linear(d_z → d_model) projection of the noisy τ latent
#   - Learned positional embedding (W positions)
#   - Sinusoidal diffusion-step embedding (→ 2-layer MLP)
#   - PhysicsEmbedding: MLP([day_features + location]) → d_model

# Transformer blocks (pre-norm, n_layers):
#   self-attention across W tokens → cross-attention to intraday physics → FFN

# Regime conditioning:
#   4 real regime labels (0=clear, 1=mixed-clear, 2=mixed-overcast, 3=overcast)
#   plus 1 null token (index 4) used during CFG unconditional passes.
#   Projection is zero-initialised so it has no effect at initialisation.
#   n_regimes is read from cfg["diffusion"]["n_regimes"]; the null token index
#   is always n_regimes (one beyond the last real class).

# Classifier-free guidance (CFG):
#   v_guided = v_uncond + guidance_scale × (v_cond − v_uncond)


import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal diffusion-step embedding
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal step embedding k → R^{d_model}, followed by 2-layer MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        assert d_model >= 6, (
            f"d_model must be >= 6 for meaningful sinusoidal frequency coverage, "
            f"got {d_model}."
        )
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """k : (B,) or (B, W) int64  →  (B, d_model) or (B, W, d_model)."""
        shape  = k.shape
        k_flat = k.reshape(-1)
        half   = self.d_model // 2
        # Standard sinusoidal positional embedding denominator is (half - 1)
        # only when the index runs 0..half-1 and you want the last freq = 1/10000.
        # The correct Vaswani et al. formula maps index i over [0, half) via
        # 10000^(i / half), so denom = half (not half-1) to avoid a spurious
        # frequency gap at the last bin that breaks long-window coverage.
        denom  = half
        freqs  = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=k.device) / denom
        )
        args  = k_flat.float().unsqueeze(1) * freqs.unsqueeze(0)
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        out   = self.proj(embed)
        return out.reshape(*shape, self.d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Physics embedding MLP
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsEmbedding(nn.Module):
    """MLP([day_features | location]) → (B, W, d_model) conditioning embedding.

    Output dimension is diffusion.d_model, not vae.latent_dim.  The two are
    independent hyper-parameters: coupling them caused a silent constraint
    (changing latent_dim silently changed the physics embedding width) and
    required a redundant phys_proj Linear in SolarDenoiser.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        dc    = cfg["diffusion"]
        in_d  = dc["day_feat_dim"] + dc["loc_feat_dim"]
        out_d = dc["d_model"]          # was: cfg["vae"]["latent_dim"] — wrong coupling
        hid   = dc["physics_embed_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(in_d, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, out_d),
        )

    def forward(
        self,
        day_features: torch.Tensor,   # (B, W, day_feat_dim)
        location:     torch.Tensor,   # (B, loc_feat_dim)
    ) -> torch.Tensor:                # (B, W, d_model)
        B, W, day_d = day_features.shape
        loc_d = location.shape[-1]
        expected_in = self.mlp[0].in_features
        actual_in   = day_d + loc_d
        assert actual_in == expected_in, (
            f"PhysicsEmbedding input dim mismatch: {actual_in} != {expected_in}. "
            f"Check diffusion.day_feat_dim and diffusion.loc_feat_dim in config."
        )
        loc = location.unsqueeze(1).expand(B, W, -1)
        inp = torch.cat([day_features, loc], dim=-1)
        return self.mlp(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Regime embedding
# ─────────────────────────────────────────────────────────────────────────────

class RegimeEmbedding(nn.Module):
    """Learnable embedding for GMM-derived 4-class regime labels.

    Vocab: 0=clear, 1=mixed-clear, 2=mixed-overcast, 3=overcast, n_regimes=null.
    The null token (index n_regimes) replaces all real labels during CFG
    unconditional passes.

    n_regimes is read from cfg["diffusion"]["n_regimes"] (default 4). The
    embedding table size is n_regimes + 1 (one null token appended at the end).

    The projection is zero-initialised so the module has no effect at
    initialisation — the denoiser gradually learns to use regime information.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        dc              = cfg["diffusion"]
        d_model         = dc["d_model"]
        embed_dim       = dc["regime_embed_dim"]
        self.n_regimes  = int(dc.get("n_regimes", 4))
        self.null_token = self.n_regimes          # always one beyond the last real class
        vocab_size      = self.n_regimes + 1      # real classes + null

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj  = nn.Linear(embed_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        regime_ids: torch.Tensor,            # (B, W) int64
        drop_mask:  Optional[torch.Tensor],  # (B,) bool — True = use null token
    ) -> torch.Tensor:                       # (B, W, d_model)
        ids = regime_ids.clone()
        if drop_mask is not None and drop_mask.any():
            null = torch.full_like(ids[0:1], self.null_token)
            ids  = torch.where(drop_mask.unsqueeze(1), null.expand_as(ids), ids)
        return self.proj(self.embed(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Intra-day physics projector (cross-attention key/value source)
# ─────────────────────────────────────────────────────────────────────────────

class IntraDayPhysicsProjector(nn.Module):
    """Project intraday physics (B, W, T_max, 3) → (B, W, T_max, d_model)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        proj_dim = cfg["diffusion"]["intraday_proj_dim"]
        d_model  = cfg["diffusion"]["d_model"]
        self.proj = nn.Sequential(
            nn.Linear(3, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, d_model),
        )

    def forward(self, intraday_phys: torch.Tensor) -> torch.Tensor:
        return self.proj(intraday_phys)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block (pre-norm, self + cross attention)
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """Pre-norm block: self-attn (W day tokens) → cross-attn (intraday physics) → FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:            torch.Tensor,
        phys_kv:      torch.Tensor,
        phys_key_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, W, D = x.shape

        assert phys_kv.shape[0] == B * W, (
            f"_TransformerBlock cross-attention: phys_kv.shape[0]={phys_kv.shape[0]} "
            f"must equal B*W={B}*{W}={B*W}. "
            f"Caller must reshape phys_kv to (B*W, T_max, d_model) before passing."
        )

        h = self.norm1(x)
        sa, _ = self.self_attn(h, h, h)
        x = x + self.drop(sa)

        h = self.norm2(x).reshape(B * W, 1, D)
        ca, _ = self.cross_attn(
            h, phys_kv, phys_kv, key_padding_mask=phys_key_pad,
        )
        x = x + self.drop(ca.reshape(B, W, D))

        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Denoiser
# ─────────────────────────────────────────────────────────────────────────────

class SolarDenoiser(nn.Module):
    """Transformer denoiser for τ-space latent diffusion.

    forward() inputs:
        tau_k         : (B, W, d_z_full)   where d_z_full = vae.latent_dim + vae.z_var_dim
        k             : (B,) or (B, W) diffusion step index
        day_features  : (B, W, day_feat_dim)
        location      : (B, loc_feat_dim)
        intraday_phys : (B, W, T_max, 3)
        valid_mask    : (B, W, T_max) bool
        drop_mask     : (B,) bool — True = drop physics conditioning (CFG)
        regime_ids    : (B, W) int64 — optional 4-class regime labels

    forward() output: v_pred (B, W, d_z_full)

    The denoiser operates entirely in the full latent space (z_flat ‖ z_var).
    It has no knowledge of the d_z / z_var_dim split — that split is only
    relevant to the VAE decoder, which slices z_flat = z_full[:, :d_z] itself.

    d_z_full is derived as:
        cfg["vae"]["latent_dim"] + cfg["vae"].get("z_var_dim", 0)
    Add vae.z_var_dim to config to enable the z_var variability latent.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        dc       = cfg["diffusion"]
        vc       = cfg["vae"]
        # d_z_full = z_flat dims + z_var dims.  z_var_dim defaults to 0 so this
        # is backward-compatible with checkpoints trained without z_var.
        d_z_full = vc["latent_dim"] + int(vc.get("z_var_dim", 0))
        d_model  = dc["d_model"]
        n_heads  = dc["n_heads"]
        n_layers = dc["n_layers"]
        dropout  = dc["dropout"]
        ffn_dim  = d_model * dc["ffn_mult"]
        W        = cfg["data"]["window_size"]

        self.d_z     = d_z_full   # kept as self.d_z for API compatibility
        self.d_model = d_model

        self.tau_proj  = nn.Linear(d_z_full, d_model)
        self.step_emb  = DiffusionStepEmbedding(d_model)
        # No step_proj: DiffusionStepEmbedding.proj already outputs d_model.
        # A second Linear(d_model, d_model) on top is a redundant identity-init
        # layer that adds parameters without semantic purpose.
        self.pos_emb   = nn.Embedding(W, d_model)
        self.register_buffer("pos_ids", torch.arange(W))

        self.phys_emb  = PhysicsEmbedding(cfg)
        # phys_proj removed: PhysicsEmbedding now outputs d_model directly.
        # The old Linear(d_z, d_model) was a redundant second projection on top
        # of PhysicsEmbedding's own final Linear(hid, d_z).

        # Learned null embeddings for CFG unconditional passes.
        # null_phys replaces the intraday cross-attn K/V stream.
        # null_day_out replaces the day_features+location conditioning.
        self.null_phys    = nn.Parameter(torch.zeros(d_model))
        self.null_day_out = nn.Parameter(torch.zeros(d_model))

        self.intraday_proj = IntraDayPhysicsProjector(cfg)

        # 4-class regime embedding with null token at index n_regimes
        self.regime_emb = RegimeEmbedding(cfg)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_z_full)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tau_k:         torch.Tensor,
        k:             torch.Tensor,
        day_features:  torch.Tensor,
        location:      torch.Tensor,
        intraday_phys: torch.Tensor,
        valid_mask:    torch.Tensor,
        drop_mask:     torch.Tensor | None = None,
        phys_kv_cache: torch.Tensor | None = None,
        key_pad_cache: torch.Tensor | None = None,
        regime_ids:    torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, W, _ = tau_k.shape

        x = self.tau_proj(tau_k)

        # Positional embedding: slice to actual W (handles shorter inference windows)
        pos_len   = min(W, self.pos_ids.shape[0])
        if pos_len < W:
            extra     = W - pos_len
            pad_ids   = torch.zeros(extra, dtype=torch.long, device=tau_k.device)
            pos_ids_w = torch.cat([self.pos_ids[:pos_len], pad_ids])
        else:
            pos_ids_w = self.pos_ids[:W]
        x = x + self.pos_emb(pos_ids_w)

        step_e = self.step_emb(k)
        if step_e.dim() == 2:
            step_e = step_e.unsqueeze(1)
        x = x + step_e

        phys_d = self.phys_emb(day_features, location)   # (B, W, d_model)

        if drop_mask is not None and drop_mask.any():
            null_day_exp = self.null_day_out.view(1, 1, -1).expand(B, W, self.d_model)
            dm     = drop_mask.view(B, 1, 1).float()
            phys_d = phys_d * (1.0 - dm) + null_day_exp * dm

        x = x + phys_d

        if regime_ids is not None:
            x = x + self.regime_emb(regime_ids, drop_mask)

        # Intra-day physics cross-attention keys/values
        cfg_drop_active = drop_mask is not None and drop_mask.any()

        if phys_kv_cache is not None and not cfg_drop_active:
            phys_kv    = phys_kv_cache
            key_pad_bw = key_pad_cache
        else:
            intraday_kv = self.intraday_proj(intraday_phys)
            T_max_kv    = intraday_phys.shape[2]
            intraday_kv = intraday_kv.reshape(B * W, T_max_kv, self.d_model)
            if cfg_drop_active:
                drop_per_day  = drop_mask.unsqueeze(1).expand(B, W).reshape(B * W)
                null_expanded = self.null_phys.view(1, 1, self.d_model).expand(
                    B * W, T_max_kv, self.d_model
                )
                keep        = (~drop_per_day).view(B * W, 1, 1).float()
                intraday_kv = intraday_kv * keep + null_expanded * (1.0 - keep)

            phys_kv    = intraday_kv
            key_pad_bw = ~valid_mask.reshape(B * W, -1)
            all_masked = key_pad_bw.all(dim=1)
            if all_masked.any():
                key_pad_bw = key_pad_bw.clone()
                key_pad_bw[all_masked, 0] = False

        for block in self.blocks:
            x = block(x, phys_kv, key_pad_bw)

        x = self.norm_out(x)
        return self.out_proj(x)

    @torch.no_grad()
    def precompute_intraday_kv(
        self,
        intraday_phys: torch.Tensor,
        valid_mask:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-project intraday physics to cross-attention K/V once before the
        denoising loop. Intraday physics are deterministic so projecting them
        inside every denoising step wastes compute.

        Returns
        -------
        phys_kv    : (B*W, T_max, d_model)
        key_pad_bw : (B*W, T_max) bool — True = padding position
        """
        B, W, T_max, _ = intraday_phys.shape
        kv = self.intraday_proj(intraday_phys).reshape(B * W, T_max, self.d_model)

        key_pad_bw = ~valid_mask.reshape(B * W, T_max)
        all_masked = key_pad_bw.all(dim=1)
        if all_masked.any():
            key_pad_bw = key_pad_bw.clone()
            key_pad_bw[all_masked, 0] = False

        return kv, key_pad_bw

    @torch.no_grad()
    def forward_cfg(
        self,
        tau_k:          torch.Tensor,
        k:              torch.Tensor,
        day_features:   torch.Tensor,
        location:       torch.Tensor,
        intraday_phys:  torch.Tensor,
        valid_mask:     torch.Tensor,
        guidance_scale: float,
        phys_kv_cache:  torch.Tensor | None = None,
        key_pad_cache:  torch.Tensor | None = None,
        regime_ids:     torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CFG inference: v_guided = v_uncond + guidance_scale · (v_cond − v_uncond).

        The unconditional pass uses drop_mask=all_True, which routes the regime
        embedding to the null token (index n_regimes) and replaces the physics
        conditioning with the learned null embeddings.
        """
        B       = tau_k.shape[0]
        no_drop  = torch.zeros(B, device=tau_k.device, dtype=torch.bool)
        all_drop = torch.ones(B,  device=tau_k.device, dtype=torch.bool)

        v_cond   = self.forward(tau_k, k, day_features, location,
                                intraday_phys, valid_mask, drop_mask=no_drop,
                                phys_kv_cache=phys_kv_cache, key_pad_cache=key_pad_cache,
                                regime_ids=regime_ids)
        v_uncond = self.forward(tau_k, k, day_features, location,
                                intraday_phys, valid_mask, drop_mask=all_drop,
                                regime_ids=regime_ids)
        return v_uncond + guidance_scale * (v_cond - v_uncond)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion training loss
# ─────────────────────────────────────────────────────────────────────────────

def diffusion_loss(
    v_pred:          torch.Tensor,
    v_target:        torch.Tensor,
    snr:             Optional[torch.Tensor] = None,
    min_snr_gamma:   float = 5.0,
    day_weights:     Optional[torch.Tensor] = None,
    day_mask:        Optional[torch.Tensor] = None,
    # ── Auxiliary losses ────────────────────────────────────────────────────
    tau0_hat:        Optional[torch.Tensor] = None,
    tau0_true:       Optional[torch.Tensor] = None,
    temporal_weight: float = 0.0,
    variance_weight: float = 0.0,
    ramp_snr_floor:  float = 1.0,
    snr_for_aux:     Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-SNR-γ weighted MSE loss on v-prediction targets (Hang et al. 2023).

    w(k) = min(SNR_k, γ) / (1 + SNR_k)

    Parameters
    ----------
    snr:              (B,) or (B, W) per-step or per-day SNR.
    day_weights:      (B, W) per-day regime emphasis weights.
                      Applied BEFORE reducing over W so that each day is weighted
                      individually by its regime.  Previous design averaged these
                      into a (B,) window-mean first, which diluted overcast weights
                      in predominantly-clear windows:
                        [13 clear + 1 overcast] → effective weight ≈ 1.11 × overcast day
                        instead of 2.5 × overcast day.
                      Per-day application corrects this.
    day_mask:         (B, W) bool — only True positions contribute to the loss.
                      Used during inpainting training to exclude context positions
                      pinned to clean tau0 (their v-target is meaningless).
    tau0_hat:         (B, W, d_z) predicted τ₀ recovered from v-prediction.
                      Required when temporal_weight > 0 or variance_weight > 0.
    tau0_true:        (B, W, d_z) ground-truth τ₀.  Required for variance_weight.
    temporal_weight:  Weight for day-to-day τ-difference MSE (ramp coherence).
                      Penalises |Δτ̂₀[t] − Δτ₀[t]|² where Δ = consecutive difference.
                      Helps the denoiser learn realistic ramp profiles between
                      clear and overcast days.  Only applied at SNR ≥ ramp_snr_floor
                      (high-quality denoising steps; skip at very high noise).
    variance_weight:  Weight for window-level τ-std matching.
                      Penalises |std(τ̂₀) − std(τ₀)|² across the W-day window.
                      Prevents the denoiser from collapsing toward the mean τ
                      (mode collapse in τ-space → all days look like mixed regime).
    ramp_snr_floor:   Minimum SNR to apply aux losses.  Aux losses on high-noise
                      steps are dominated by noise rather than signal and add
                      noise to the gradient.  Default 1.0 (√ᾱ ≈ √(1−ᾱ)).
    snr_for_aux:      SNR tensor for aux-loss gating (same shape as snr).

    Returns
    -------
    loss:      Total scalar loss (MSE + temporal + variance).
    l_temporal: Temporal coherence aux loss (zero tensor if weight=0).
    l_variance: Distribution variance aux loss (zero tensor if weight=0).
    """
    err = (v_pred - v_target) ** 2                    # (B, W, d_z)
    err_per_day = err.mean(dim=2)                     # (B, W)

    # ── Min-SNR-γ weighting ─────────────────────────────────────────────────
    if snr is not None:
        w_snr = torch.clamp(snr, max=min_snr_gamma) / (1.0 + snr)
        if w_snr.dim() == 1:
            w_snr = w_snr.view(-1, 1)                # broadcast over W
        err_per_day = err_per_day * w_snr

    # ── Per-day regime weights ───────────────────────────────────────────────
    if day_weights is not None:
        err_per_day = err_per_day * day_weights

    # ── Day mask (inpainting context exclusion) ──────────────────────────────
    if day_mask is not None:
        mask_f     = day_mask.float()
        n_active   = mask_f.sum(dim=1).clamp(min=1.0)
        per_sample = (err_per_day * mask_f).sum(dim=1) / n_active
    else:
        per_sample = err_per_day.mean(dim=1)

    mse_loss = per_sample.mean()

    # ── Temporal coherence auxiliary loss ────────────────────────────────────
    # |Δτ̂₀[t] − Δτ₀[t]|²  where Δ = first-order consecutive difference across W.
    # Penalises errors in day-to-day transitions (ramp reproduction).
    # Only applied at steps where SNR ≥ ramp_snr_floor.
    l_temporal = torch.zeros(1, device=v_pred.device)
    if temporal_weight > 0.0 and tau0_hat is not None:
        snr_gate = snr_for_aux if snr_for_aux is not None else snr
        if snr_gate is not None:
            snr_scalar = snr_gate.reshape(-1).mean()
            apply_temporal = (snr_scalar >= ramp_snr_floor)
        else:
            apply_temporal = True
        if apply_temporal:
            d_hat  = tau0_hat[:, 1:, :] - tau0_hat[:, :-1, :]   # (B, W-1, d_z)
            d_true = tau0_true[:, 1:, :] - tau0_true[:, :-1, :]
            l_temporal = temporal_weight * (d_hat - d_true).pow(2).mean()

    # ── Distribution variance auxiliary loss ─────────────────────────────────
    # |std(τ̂₀, dim=W) − std(τ₀, dim=W)|²  per batch element.
    # Penalises variance collapse in τ-space (all days converging toward mean τ).
    l_variance = torch.zeros(1, device=v_pred.device)
    if variance_weight > 0.0 and tau0_hat is not None and tau0_true is not None:
        std_hat  = tau0_hat.std(dim=1)       # (B, d_z)
        std_true = tau0_true.std(dim=1)      # (B, d_z)
        l_variance = variance_weight * (std_hat - std_true).pow(2).mean()

    loss = mse_loss + l_temporal + l_variance
    return loss, l_temporal.detach(), l_variance.detach()


# ─────────────────────────────────────────────────────────────────────────────
# EMA of model weights
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """EMA of model parameters: shadow ← decay·shadow + (1−decay)·param."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA shadow weights into model parameters."""
        for k, v in model.named_parameters():
            v.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module, original: Dict) -> None:
        """Restore raw model weights from a snapshot taken before copy_to()."""
        for k, v in model.named_parameters():
            v.data.copy_(original[k])