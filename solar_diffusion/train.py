#
# solar_diffusion/train.py
# -------------------------
# Training loops for Stage 1 (AE) and Stage 2 (Diffusion).
# 
# Stage 1 logs (every epoch):
#   train_recon / val_recon        Masked L1. Checkpoint criterion.
#   train_sep / fired_pct          Regime separation loss. Target: fired > 0% within 5 epochs.
#   val_recon_clear/mixed/overcast Per-regime reconstruction.
#   latent_sep_dist                L2 distance between clear/cloudy centroids. Target > margin.
#   ramp_mae                       Intraday shape fidelity.
# 
# Stage 2 logs (every epoch):
#   train_loss / val_loss / ema    Diffusion loss.
#   cloudy/clear ratio             Per-day ratio. Target 1-4 after warmup.
#   tau_separation                 cloudy_mean - clear_mean in tau-space. Target > 0.20.
#   trans_error_fro                Transition matrix error. Target < 0.15.
#

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from solar_diffusion.denoiser import EMA, SolarDenoiser, diffusion_loss
from solar_diffusion.optical_depth import LatentTauTransform, TauNoiseSchedule
from solar_diffusion.vae import SolarVAE, ae_loss, regime_separation_loss, latent_variance_penalty
from solar_diffusion.training_controller import TrainingController, ctrl_actions_to_record

log = logging.getLogger(__name__)


# =============================================================================
# Basic utilities
# =============================================================================

def get_device(cfg: Dict) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    epoch: int,
    metric: float,
    extra: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":           epoch,
        "metric":          metric,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info("Checkpoint saved -> %s  (epoch=%d  metric=%.5f)", path, epoch, metric)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    log.info("Checkpoint loaded <- %s  (epoch=%d)", path, ckpt.get("epoch", -1))
    return ckpt


def _get_obs_phys(
    batch: Dict,
    device: torch.device,
    B: int,
    W: int,
    T_max: int,
) -> Optional[torch.Tensor]:
    #Extract and reshape obs_phys from a batch dict for the AE encoder.
    # Returns (B*W, T_max, N_obs) or None.
    #
    obs = batch.get("obs_phys", None)
    if obs is None:
        return None
    obs = obs.to(device)
    N_obs = obs.shape[-1]
    if N_obs == 0:
        return None
    return obs.reshape(B * W, T_max, N_obs)


def _compute_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq ** 0.5


def _compute_param_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        total_sq += p.detach().float().norm(2).item() ** 2
    return total_sq ** 0.5


def _compute_per_layer_grad_norms(model: nn.Module) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        sq = sum(
            p.grad.detach().float().norm(2).item() ** 2
            for p in params
            if p.grad is not None
        )
        if sq > 0:
            norms[name] = sq ** 0.5
    return norms


def _compute_ema_param_diff(model: nn.Module, ema: "EMA") -> float:
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if name in ema.shadow:
            total += (p.detach().float() - ema.shadow[name].float()).abs().mean().item()
            count += 1
    return total / max(count, 1)


def _log_per_layer_grads(model: nn.Module, epoch: int, stage: str) -> None:
    norms = _compute_per_layer_grad_norms(model)
    if not norms:
        return
    sorted_layers = sorted(norms.items(), key=lambda x: -x[1])
    log.info("[%s] Epoch %d -- Per-layer gradient norms (top 10):", stage, epoch)
    for name, norm in sorted_layers[:10]:
        log.info("  %-55s  %.4e", name, norm)
    dead = [n for n in norms if norms[n] == 0.0]
    if dead:
        log.warning("[%s] Epoch %d -- %d dead layers (zero grad): %s",
                    stage, epoch, len(dead), dead[:5])


def _append_json_log(log_path: Path, record: Dict) -> None:
    #Append one epoch record to a newline-delimited JSON log file.
    # 
    # Each line is a valid JSON object.  The file is human-readable and can be
    # loaded for analysis with:
    #     import json
    #     records = [json.loads(l) for l in open("vae_training_log.jsonl")]
    #     import pandas as pd
    #     df = pd.DataFrame(records)
    # 
    # Writing is append-only so a resumed run adds to the same file rather than
    # overwriting it.  Safe for concurrent readers (each line is atomic).
    #
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# =============================================================================
# Shared regime helpers
# =============================================================================

def _regime_from_batch(batch: Dict, device: torch.device) -> torch.Tensor:
    #Return batch["regime"] as a (B*W,) int64 tensor on device.
    # 
    # P8: all regime classification is done by RegimeGMM in dataset.py and stored
    # in batch["regime"]. This helper centralises the reshape so every call site
    # (VAE epoch, diffusion epoch, diagnostics) reads from the same source.
    #
    B, W = batch["regime"].shape
    return batch["regime"].to(device).reshape(B * W)


def _day_mean_k_tensor(
    k_star_bw: torch.Tensor,
    valid_mask_bw: torch.Tensor,
) -> torch.Tensor:
    #Mean sunlit K* per day.  Returns (N,).
    n_valid = valid_mask_bw.float().sum(dim=1).clamp(min=1.0)
    return (k_star_bw * valid_mask_bw.float()).sum(dim=1) / n_valid


def _day_std_k_tensor(
    k_star_bw: torch.Tensor,
    valid_mask_bw: torch.Tensor,
) -> torch.Tensor:
    #Intraday std of sunlit K* per day.  Returns (N,).
    # 
    # Uses Bessel-corrected std (n-1).  Days with <2 valid timesteps return 0.
    # 
    # Fix: clamp(min=1.0) is used for the mean (safe denominator) but we need
    # the TRUE n_valid for the Bessel denominator - not clamped to 2.  Days with
    # 0 or 1 sunlit timestep are masked out and return 0.0 rather than a garbage
    # value (previously clamp(min=2) gave n-1=1 but a wrong mean, producing a
    # non-zero std from a single sample that contaminates RegimeGMM features and
    # sampling weights).
    #
    mask_f  = valid_mask_bw.float()
    n_valid = mask_f.sum(dim=1)                             # (N,) true count, no clamp
    # Safe mean: use clamp(min=1) only for the mean denominator
    mean_k  = (k_star_bw * mask_f).sum(dim=1) / n_valid.clamp(min=1.0)
    sq_diff = ((k_star_bw - mean_k.unsqueeze(1)) ** 2) * mask_f
    # Bessel denominator: (n-1) for n>=2, else 0 -> std = 0
    bessel  = (n_valid - 1.0).clamp(min=0.0)               # (N,) - 0 when n_valid < 2
    # Avoid division by zero: where bessel==0 the result is 0
    std     = torch.where(
        bessel > 0,
        (sq_diff.sum(dim=1) / bessel).sqrt(),
        torch.zeros_like(bessel),
    )
    return std


# =============================================================================
# Stage 1 - AE diagnostics
# =============================================================================

@torch.no_grad()
def _vae_val_diagnostics(
    model: SolarVAE,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict,
) -> Dict[str, float]:
    #Full val-set pass collecting all AE diagnostics.
    # 
    # P8: per-regime buckets use batch["regime"] (GMM labels from dataset.py)
    # instead of recomputing mean_k thresholds here.
    # 
    # Returns
    # -------
    # recon_clear / recon_mixed / recon_overcast  per-regime masked L1
    # ramp_mae      MAE on timesteps where |delta K*| > 0.05
    # latent_mean / latent_std / latent_min / latent_max
    # latent_sep_dist   L2 distance between clear and cloudy centroids in z-space
    #
    k_max         = float(cfg["physics"]["k_max"])
    ramp_diag_thr = float(cfg["vae"]["ramp_diag_threshold"])
    n_regimes     = int(cfg["diffusion"].get("n_regimes", 4))
    # C7: read intraday_physics_dim from config - never hardcode 3.
    # All intraday_phys reshapes in this function use _phys_dim so adding
    # a 4th physics channel only requires a config change.
    _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))

    recon_sum  = [0.0] * n_regimes
    recon_n    = [0]   * n_regimes
    ramp_sum   = [0.0] * n_regimes
    ramp_n     = [0]   * n_regimes
    all_z:       List[np.ndarray] = []
    all_regimes: List[np.ndarray] = []
    total_clamp_n = 0
    total_valid_n = 0

    model.eval()
    for batch in val_loader:
        B, W, T_max = batch["k_star"].shape
        k_star_bw = batch["k_star"].to(device).reshape(B * W, T_max)
        phys_bw   = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
        mask_bw   = batch["valid_mask"].to(device).reshape(B * W, T_max)
        obs_bw    = _get_obs_phys(batch, device, B, W, T_max)
        # P8: read GMM labels from batch instead of recomputing with thresholds
        regime_bw = _regime_from_batch(batch, device)   # (B*W,) int64

        k_hat_bw, z_bw = model(k_star_bw, phys_bw, mask_bw, obs_bw)
        k_combined_bw  = k_hat_bw
        # M1: slice only z_flat dims (first latent_dim) for diagnostics.
        # model() returns z_full = cat(z_flat, z_var) with shape (B*W, latent_dim+z_var_dim).
        # z_var dims have different statistics (narrow range, systematic bias) and
        # contaminate latent_mean, latent_std, dead_dims, and centroid separation.
        _d_z_flat = int(cfg["vae"]["latent_dim"])
        all_z.append(z_bw[:, :_d_z_flat].cpu().float().numpy())
        all_regimes.append(regime_bw.cpu().numpy())

        mask_f  = mask_bw.float()
        n_valid = mask_f.sum(dim=1).clamp(min=1.0)
        l1_per  = ((k_combined_bw - k_star_bw).abs() * mask_f).sum(dim=1) / n_valid

        total_clamp_n += int(((k_combined_bw > 0.99 * k_max) & mask_bw).sum().item())
        total_valid_n += int(mask_bw.sum().item())

        delta_real  = (k_star_bw[:, 1:]     - k_star_bw[:, :-1]).abs()
        delta_hat   = (k_combined_bw[:, 1:] - k_combined_bw[:, :-1]).abs()
        consec_ok   = mask_bw[:, :-1] & mask_bw[:, 1:]
        ramp_mask_t = consec_ok & (delta_real > ramp_diag_thr)

        for i in range(B * W):
            r = int(regime_bw[i].item())
            recon_sum[r] += float(l1_per[i].item())
            recon_n[r]   += 1
            if ramp_mask_t[i].any():
                ramp_sum[r] += float(
                    (delta_hat[i] - delta_real[i]).abs()[ramp_mask_t[i]].sum().item()
                )
                ramp_n[r]   += int(ramp_mask_t[i].sum().item())

    z_np       = np.concatenate(all_z,       axis=0)
    regime_np  = np.concatenate(all_regimes, axis=0)

    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]
    out: Dict[str, float] = {}
    for i, nm in enumerate(regime_names):
        out[f"recon_{nm}"]    = recon_sum[i] / recon_n[i] if recon_n[i] > 0 else float("nan")
        out[f"recon_n_{nm}"]  = float(recon_n[i])
        out[f"ramp_mae_{nm}"] = ramp_sum[i]  / ramp_n[i] if ramp_n[i] > 0 else float("nan")

    out["ramp_mae"]       = sum(ramp_sum) / max(sum(ramp_n), 1)
    out["latent_mean"]    = float(z_np.mean())
    out["latent_std"]     = float(z_np.std())
    out["latent_min"]     = float(z_np.min())
    out["latent_max"]     = float(z_np.max())

    per_dim_std               = z_np.std(axis=0)
    dead_std_thr = float(cfg["vae"].get("dead_dim_std_threshold", 0.05))
    hot_std_thr  = float(cfg["vae"].get("hot_dim_std_threshold",  4.0))
    out["latent_dim_std_min"] = float(per_dim_std.min())
    out["latent_dim_std_max"] = float(per_dim_std.max())
    out["latent_n_dead_dims"] = float((per_dim_std < dead_std_thr).sum())
    out["latent_n_hot_dims"]  = float((per_dim_std > hot_std_thr).sum())
    out["decoder_clamp_rate"] = total_clamp_n / max(total_valid_n, 1)

    # Regime centroid separations using GMM labels
    is_clear    = (regime_np == 0)
    is_overcast = (regime_np == n_regimes - 1)   # last class is always the most overcast
    is_cloudy   = ~is_clear

    out["latent_n_clear"]    = float(is_clear.sum())
    out["latent_n_overcast"] = float(is_overcast.sum())
    for i, nm in enumerate(regime_names):
        out[f"latent_n_{nm}"] = float((regime_np == i).sum())

    if is_clear.sum() >= 2 and is_cloudy.sum() >= 2:
        mu_c  = z_np[is_clear].mean(axis=0)
        mu_cl = z_np[is_cloudy].mean(axis=0)
        out["latent_sep_dist"]    = float(np.linalg.norm(mu_c - mu_cl))
        dim_sep                    = np.abs(mu_c - mu_cl)
        out["latent_sep_top_dim"] = int(dim_sep.argmax())
        out["latent_sep_top_val"] = float(dim_sep.max())
    else:
        out["latent_sep_dist"]    = float("nan")
        out["latent_sep_top_dim"] = -1
        out["latent_sep_top_val"] = float("nan")

    if is_clear.sum() >= 2 and is_overcast.sum() >= 2:
        out["latent_sep_clear_overcast"] = float(
            np.linalg.norm(z_np[is_clear].mean(axis=0) - z_np[is_overcast].mean(axis=0))
        )
    else:
        out["latent_sep_clear_overcast"] = float("nan")

    # All adjacent-class separations
    for i in range(n_regimes):
        for j in range(i + 1, n_regimes):
            a_mask = (regime_np == i)
            b_mask = (regime_np == j)
            key = f"latent_sep_{regime_names[i]}_{regime_names[j]}"
            if a_mask.sum() >= 2 and b_mask.sum() >= 2:
                out[key] = float(np.linalg.norm(
                    z_np[a_mask].mean(axis=0) - z_np[b_mask].mean(axis=0)
                ))
            else:
                out[key] = float("nan")

    return out


def _log_vae_diagnostics(diag: Dict[str, float], val_recon: float, cfg: Dict) -> None:
    alpha        = cfg["vae"]["dynamic_margin_alpha"]
    n_regimes    = int(cfg["diffusion"].get("n_regimes", 4))
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]

    # All warning thresholds come from config - no magic numbers here.
    mixed_recon_warn  = float(cfg["vae"]["mixed_recon_warn_ratio"])
    mixed_recon_err   = float(cfg["vae"]["mixed_recon_err_ratio"])
    oc_recon_err      = float(cfg["vae"]["oc_recon_err_ratio"])
    ramp_warn         = float(cfg["vae"]["ramp_warn_ratio"])
    latent_mean_warn  = float(cfg["vae"]["latent_mean_warn"])
    latent_std_warn   = float(cfg["vae"]["latent_std_warn"])
    latent_max_warn   = float(cfg["vae"]["latent_max_warn"])
    decoder_clamp_warn= float(cfg["vae"]["decoder_clamp_warn"])
    dead_std_thr      = float(cfg["vae"]["dead_dim_std_threshold"])
    hot_std_thr       = float(cfg["vae"]["hot_dim_std_threshold"])

    def _fmt(nm: str) -> str:
        v = diag.get(f"recon_{nm}", float("nan"))
        n = int(diag.get(f"recon_n_{nm}", 0))
        return "N/A (N=0)" if np.isnan(v) else f"{v:.5f} (N={n})"

    def _ramp(nm: str) -> str:
        v = diag.get(f"ramp_mae_{nm}", float("nan"))
        return "N/A" if np.isnan(v) else f"{v:.5f}"

    log.info("  [REGIME RECON]    " + "  ".join(f"{nm}={_fmt(nm)}" for nm in regime_names))

    rc = diag.get("recon_clear", float("nan"))
    ro = diag.get(f"recon_{regime_names[-1]}", float("nan"))

    for nm in regime_names[1:]:
        rv = diag.get(f"recon_{nm}", float("nan"))
        if not np.isnan(rv) and not np.isnan(rc) and rc > 0:
            gap = rv / rc
            if gap > mixed_recon_err:
                log.warning("  [!!] %s recon %.2fx clear - AE fails on partial cloudiness.", nm, gap)
            elif gap > mixed_recon_warn:
                log.warning("  [!]  %s recon %.2fx clear - watch for further widening.", nm, gap)

    if not np.isnan(rc) and not np.isnan(ro) and ro > oc_recon_err * max(rc, 1e-8):
        log.warning("  [!!] %s recon %.1fx > clear - AE underfits cloudy profiles.", regime_names[-1], ro / max(rc, 1e-8))
    if np.isnan(ro):
        log.warning("  [!] %s bucket empty (N=0) - val set has no overcast days.", regime_names[-1])

    log.info(
        "  [RAMP MAE]        overall=%.5f  %s  (vs val_recon=%.5f  ratio=%.2f)",
        diag["ramp_mae"],
        "  ".join(f"{nm}={_ramp(nm)}" for nm in regime_names),
        val_recon, diag["ramp_mae"] / max(val_recon, 1e-8),
    )
    log.info(
        "  [LATENT GLOBAL]   mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        diag["latent_mean"], diag["latent_std"], diag["latent_min"], diag["latent_max"],
    )
    log.info(
        "  [LATENT DIMS]     dim_std: min=%.3f  max=%.3f  dead(std<%.2f)=%.0f  hot(std>%.1f)=%.0f",
        diag["latent_dim_std_min"], diag["latent_dim_std_max"],
        dead_std_thr, diag["latent_n_dead_dims"],
        hot_std_thr,  diag["latent_n_hot_dims"],
    )
    log.info("  [DECODER]         clamp_rate=%.3f  (target <%.2f)", diag["decoder_clamp_rate"], decoder_clamp_warn)

    sep_dist = diag.get("latent_sep_dist", float("nan"))
    # SAFETY CHECK: if sep_dist is nan, regime_ids are not reaching the loss.
    # This means separation_loss returns 0 silently every batch - the encoder
    # will collapse with no resistance.  Most likely cause: batch["regime"] is
    # absent from the dataset or min_samp is never satisfied.
    if np.isnan(sep_dist):
        log.info("  [LATENT SEP]      N/A (insufficient samples in one regime bucket)")
    else:
        n_str = "  ".join(f"{nm}=%.0f" % diag.get(f"latent_n_{nm}", 0) for nm in regime_names)
        log.info(
            "  [LATENT SEP]      clear-cloudy=%.4f  top-dim=%d (|Deltamu|=%.3f)  N: %s",
            sep_dist, int(diag.get("latent_sep_top_dim", -1)),
            diag.get("latent_sep_top_val", float("nan")), n_str,
        )
        sep_pairs = {k: v for k, v in diag.items()
                     if k.startswith("latent_sep_") and "top" not in k and k != "latent_sep_dist"}
        if sep_pairs:
            log.info(
                "  [LATENT SEP ALL]  %s  (alpha=%.2f  mixed_scale=%.2f)",
                "  ".join(f"{k.replace('latent_sep_','')}=%.4f" % v for k, v in sep_pairs.items()),
                alpha, cfg["vae"]["mixed_margin_scale"],
            )

    if diag["ramp_mae"] > ramp_warn * max(val_recon, 1e-8):
        log.warning("  [!] ramp_mae=%.5f is %.1fx val_recon - raise spectral_weight.",
                    diag["ramp_mae"], diag["ramp_mae"] / max(val_recon, 1e-8))
    if abs(diag["latent_mean"]) > latent_mean_warn:
        log.warning("  [!] latent_mean=%.3f far from 0 - tau-transform bias. Re-fit.", diag["latent_mean"])
    if diag["latent_std"] > latent_std_warn:
        log.warning("  [!] latent_std=%.3f >> 1 - re-fit tau stats.", diag["latent_std"])
    if abs(diag["latent_max"]) > latent_max_warn:
        log.warning("  [!] latent_max=%.2f - overflow risk in tau-space.", diag["latent_max"])
    if diag["latent_n_dead_dims"] > 0:
        _dead_pct = 100.0 * diag["latent_n_dead_dims"] / max(float(cfg["vae"]["latent_dim"]), 1)
        _dead_n   = int(diag["latent_n_dead_dims"])
        if _dead_pct >= 20.0:
            # H4: >=20% dead dims is a critical failure requiring action.
            # Root cause: z_std collapse maps multiple days to the same point,
            # leaving many dims with near-zero variance permanently.
            # Action: raise z_var_weight so the variance penalty pushes harder.
            log.warning(
                "  [!!] %.0f/%.0f dead latent dims (%.0f%%) - "
                "CRITICAL: latent collapse. z_std has likely dropped below target. "
                "ACTION: raise z_var_weight (currently %.1f) by 1.5x or check "
                "z_std_target=%.2f is not too high for current sep_weight=%.1f.",
                _dead_n, float(cfg["vae"]["latent_dim"]), _dead_pct,
                float(cfg["vae"].get("z_var_weight", 1.0)),
                float(cfg["vae"].get("z_std_target", 0.60)),
                float(cfg["vae"].get("regime_sep_weight", 1.5)),
            )
        elif _dead_pct >= 10.0:
            log.warning(
                "  [!] %.0f dead latent dims (%.0f%%) - wasted capacity. "
                "Consider raising z_var_weight if this persists >5 epochs. "
                "Current z_var_weight=%.1f  z_std_target=%.2f.",
                _dead_n, _dead_pct,
                float(cfg["vae"].get("z_var_weight", 1.0)),
                float(cfg["vae"].get("z_std_target", 0.60)),
            )
        else:
            log.warning("  [!] %.0f dead latent dims (%.0f%%) - wasted capacity.", _dead_n, _dead_pct)
    if diag["latent_n_hot_dims"] > 0:
        log.warning("  [!] %.0f hot latent dims - tau-transform overflow risk.", diag["latent_n_hot_dims"])
    if diag["decoder_clamp_rate"] > decoder_clamp_warn:
        log.warning("  [!] decoder_clamp_rate=%.3f > %.2f - sigmoid saturating.",
                    diag["decoder_clamp_rate"], decoder_clamp_warn)


def _vae_action_guide(best_diag: Dict[str, float], best_val_recon: float, cfg: Dict) -> None:
    sw     = cfg["vae"]["spectral_weight"]
    ld     = cfg["vae"]["latent_dim"]
    sw_sep = cfg["vae"]["regime_sep_weight"]
    alpha  = cfg["vae"]["dynamic_margin_alpha"]
    n_regimes    = int(cfg["diffusion"].get("n_regimes", 4))
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]
    log.info("=" * 72)
    log.info("STAGE 1 COMPLETE -- ACTION GUIDE")
    log.info("=" * 72)
    log.info("  Best val_recon        : %.5f", best_val_recon)

    def _afmt(nm: str) -> str:
        v = best_diag.get(f"recon_{nm}", float("nan"))
        n = int(best_diag.get(f"recon_n_{nm}", 0))
        return "N/A (N=0)" if np.isnan(v) else f"{v:.5f} (N={n})"

    log.info("  Per-regime recon      : " + "  ".join(f"{nm}={_afmt(nm)}" for nm in regime_names))
    log.info("  Ramp MAE              : %.5f  (ratio vs recon: %.2f)",
             best_diag.get("ramp_mae", 0),
             best_diag.get("ramp_mae", 0) / max(best_val_recon, 1e-8))
    log.info("  Latent mean/std/max   : %.3f / %.3f / %.3f",
             best_diag.get("latent_mean", 0), best_diag.get("latent_std", 0),
             best_diag.get("latent_max", 0))
    dead_std_thr = float(cfg["vae"].get("dead_dim_std_threshold", 0.05))
    hot_std_thr  = float(cfg["vae"].get("hot_dim_std_threshold",  4.0))
    log.info("  Dead dims / Hot dims  : %.0f / %.0f  (std<%.2f / std>%.1f)",
             best_diag.get("latent_n_dead_dims", 0), best_diag.get("latent_n_hot_dims", 0),
             dead_std_thr, hot_std_thr)
    log.info("  Decoder clamp rate    : %.3f  (target <0.05)",
             best_diag.get("decoder_clamp_rate", 0))
    def _ramp_fmt(nm: str) -> str:
        v = best_diag.get(f"ramp_mae_{nm}", float("nan"))
        return "N/A" if np.isnan(v) else f"{v:.5f}"
    log.info("  Ramp MAE per regime   : " + "  ".join(f"{nm}={_ramp_fmt(nm)}" for nm in regime_names))
    sep_dist = best_diag.get("latent_sep_dist", float("nan"))
    if np.isnan(sep_dist):
        log.info("  Latent sep dist       : N/A")
    else:
        log.info("  Latent sep dist       : %.4f  (dynamic_margin_alpha=%.2f)", sep_dist, alpha)

    mixed_recon_err = float(cfg["vae"]["mixed_recon_err_ratio"])
    oc_recon_err    = float(cfg["vae"]["oc_recon_err_ratio"])
    ramp_warn       = float(cfg["vae"]["ramp_warn_ratio"])

    issues = []
    rc = best_diag.get("recon_clear", 0.0)
    ro = best_diag.get(f"recon_{regime_names[-1]}", float("nan"))
    rm_ratio = best_diag.get("ramp_mae", 0.0) / max(best_val_recon, 1e-8)

    if np.isnan(ro):
        issues.append(
            f"  WARNING: {regime_names[-1]} bucket was empty (N=0) throughout training. "
            "Add stations with more cloudy days."
        )
    elif ro > oc_recon_err * max(rc, 1e-8):
        issues.append(
            f"  RAISE spectral_weight {sw:.2f} -> {min(sw+0.2, 0.6):.2f} "
            f"(or RAISE latent_dim {ld} -> {ld+16}): "
            f"{regime_names[-1]} recon is {ro/max(rc,1e-8):.1f}x clear recon."
        )
    for nm in regime_names[1:]:
        rv = best_diag.get(f"recon_{nm}", float("nan"))
        if not np.isnan(rv) and rc > 0 and rv > mixed_recon_err * rc:
            issues.append(
                f"  RAISE spectral_weight {sw:.2f} -> {min(sw+0.2, 0.6):.2f} "
                f"(or RAISE latent_dim {ld} -> {ld+16}): "
                f"{nm} recon is {rv/max(rc,1e-8):.1f}x clear recon."
            )
    ramp_warn    = float(cfg["vae"]["ramp_warn_ratio"])
    sep_dist_min = float(cfg["diagnostics"].get("latent_sep_dist_min_warn", 0.5))
    tau_sep_min  = float(cfg["diagnostics"]["tau_separation_min_to_proceed"])
    latent_std_w = float(cfg["vae"]["latent_std_warn"])
    latent_max_w = float(cfg["vae"]["latent_max_warn"])
    clamp_w      = float(cfg["vae"]["decoder_clamp_warn"])

    if rm_ratio > ramp_warn:
        issues.append(
            f"  RAISE spectral_weight {sw:.2f} -> {min(sw+0.2, 0.6):.2f}: "
            f"ramp_mae is {rm_ratio:.1f}x val_recon."
        )
    if not np.isnan(sep_dist) and sep_dist < sep_dist_min:
        issues.append(
            f"  RAISE regime_sep_weight {sw_sep:.2f} -> {min(sw_sep*3.0, 4.0):.2f}: "
            f"latent_sep_dist={sep_dist:.4f} is very small - tau_separation will be < {tau_sep_min:.2f}."
        )
    if best_diag.get("latent_std", 1.0) > latent_std_w:
        issues.append(f"  Re-fit LatentTauTransform after AE converges (latent_std > {latent_std_w:.1f}).")
    if best_diag.get("latent_max", 0.0) > latent_max_w:
        issues.append(f"  Watch for tau overflow - latent_max > {latent_max_w:.0f}.")
    if best_diag.get("latent_n_dead_dims", 0) > 0:
        issues.append(
            f"  {int(best_diag['latent_n_dead_dims'])} dead latent dims - "
            f"reduce latent_dim {ld} -> {max(ld//2, 16)} or raise encoder_dropout."
        )
    if best_diag.get("decoder_clamp_rate", 0.0) > clamp_w:
        issues.append(
            f"  decoder_clamp_rate={best_diag['decoder_clamp_rate']:.3f} > {clamp_w:.2f} - "
            f"sigmoid saturating. Verify k_max={cfg['physics']['k_max']} matches data range."
        )

    if issues:
        log.warning("  RECOMMENDED CHANGES:")
        for msg in issues:
            log.warning(msg)
    else:
        log.info("  Everything looks healthy. Proceed to Stage 2.")
    log.info(
        "  NOTE: After Stage 1, run build_latent_cache + fit_latent_tau_transform "
        "and verify tau_separation > 0.20 before starting Stage 2."
    )
    log.info("=" * 72)


# =============================================================================
# Stage 2 - Diffusion diagnostics
# =============================================================================

def _log_regime_frequencies(
    loader: DataLoader,
    cfg: Dict,
    label: str = "Val",
) -> Dict[str, float]:
    #Log regime frequencies from batch GMM labels. No thresholds used.
    n_regimes    = int(cfg["diffusion"].get("n_regimes", 4))
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]
    overcast_warn = float(cfg["diagnostics"].get("overcast_min_frac_warn", 0.02))

    counts = [0] * n_regimes
    total  = 0
    for batch in loader:
        B, W = batch["regime"].shape
        labels = batch["regime"].reshape(B * W).numpy()
        for r in labels:
            idx = int(r)
            if 0 <= idx < n_regimes:
                counts[idx] += 1
            total += 1
    total = max(total, 1)
    frac  = [c / total for c in counts]
    log.info(
        "%s regime frequencies | %s  (N=%d days)",
        label,
        "  ".join(f"{nm}={100*f:.1f}%" for nm, f in zip(regime_names, frac)),
        total,
    )
    oc_frac = frac[-1]
    if oc_frac < overcast_warn:
        log.warning(
            "[!] %s set has only %.1f%% %s days - val_loss_%s will be noisy.",
            label, 100.0 * oc_frac, regime_names[-1], regime_names[-1],
        )
    return {f"frac_{nm}": f for nm, f in zip(regime_names, frac)}


@torch.no_grad()
def _diff_val_diagnostics(
    model: SolarDenoiser,
    vae: SolarVAE,
    val_loader: DataLoader,
    schedule: TauNoiseSchedule,
    latent_transform: LatentTauTransform,
    device: torch.device,
    cfg: Dict,
) -> Dict[str, float]:
    #Full val-set diagnostic pass for Stage 2.
    # 
    # Returns per-day three-way regime val MSE, SNR-stratified MSE, tau statistics.
    # 
    # Regime classification is done per individual day (not per window).
    # A window of W days contributes W separate day-level regime observations to
    # the loss buckets.  Window-level classification caused loss_overcast=0.0 and
    # loss_clear=0.0 permanently because window-mean K* almost never falls below
    # the overcast threshold when most days in a window are clear.
    #
    dc        = cfg["diffusion"]
    pred_type = dc["prediction_type"]
    n_regimes    = int(dc.get("n_regimes", 4))
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]
    # SNR stratification boundaries from config (not hardcoded 1.0 / 10.0)
    snr_lo = float(cfg["diagnostics"].get("snr_low_boundary",  1.0))
    snr_hi = float(cfg["diagnostics"].get("snr_high_boundary", 10.0))

    regime_mse = [0.0] * n_regimes
    regime_n   = [0]   * n_regimes
    snr_mse  = {"low": 0.0, "mid": 0.0, "high": 0.0}
    snr_n    = {"low": 0,   "mid": 0,   "high": 0  }
    all_tau_means: List[float]       = []
    # Per-class tau buckets (n_regimes buckets + index 0 for clear, -1 alias for overcast)
    tau_by_class: List[List[float]] = [[] for _ in range(n_regimes)]

    model.eval()
    vae.eval()
    for batch in val_loader:
        B, W, T_max = batch["k_star"].shape
        # C7: physics dim from config
        _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))
        k_bw   = batch["k_star"].to(device).reshape(B * W, T_max)
        ph_bw  = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
        ma_bw  = batch["valid_mask"].to(device).reshape(B * W, T_max)
        obs_bw = _get_obs_phys(batch, device, B, W, T_max)
        # P8: GMM regime labels - (B, W) -> reshape below as needed
        regime_bw_np = batch["regime"].numpy().reshape(B, W)   # (B, W) int

        z, _ = vae.encode(k_bw, ph_bw, ma_bw, obs_bw)
        z     = z.reshape(B, W, -1)
        tau0 = latent_transform.to_tau_space(z)

        # Tau statistics reported per-day (mean over d_z), tracked per class
        tau0_day_mean = tau0.mean(dim=2).cpu().numpy()   # (B, W)
        for b in range(B):
            for d in range(W):
                tm = float(tau0_day_mean[b, d])
                all_tau_means.append(tm)
                r = int(regime_bw_np[b, d])
                if 0 <= r < n_regimes:
                    tau_by_class[r].append(tm)

        if bool(dc.get("per_day_k_sampling", True)):
            k_step = torch.randint(0, dc["num_steps"], (B, W), device=device)
        else:
            k_step = torch.randint(0, dc["num_steps"], (B,), device=device)
        noise  = torch.randn_like(tau0)
        tau_k, noise_used = schedule.q_sample(tau0, k_step, noise)

        day_feat   = batch["day_features"].to(device)
        location   = batch["location"].to(device)
        intra_phys = batch["intraday_phys"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        drop_mask  = torch.zeros(B, device=device, dtype=torch.bool)

        v_pred = model(tau_k, k_step, day_feat, location,
                       intra_phys, valid_mask, drop_mask=drop_mask,
                       regime_ids=batch["regime"].to(device))

        if pred_type == "v":
            target = schedule.get_v_target(tau0, noise_used, k_step)
        elif pred_type == "eps":
            target = noise_used
        else:
            target = tau0

        # Per-day MSE: (B, W) - mean over d_z per day
        mse_day = (v_pred - target).pow(2).mean(dim=2)   # (B, W) - stays on GPU briefly

        # SNR per day or per window depending on k sampling mode
        if k_step.dim() == 2:
            ab_k   = schedule.alpha_bars[k_step.reshape(-1).long()].reshape(B, W)
            snr_np = ab_k.cpu().numpy()
            snr_np = snr_np / (1.0 - snr_np + 1e-8)
        else:
            ab_k   = schedule.alpha_bars[k_step.long()].cpu().numpy()
            snr_np = (ab_k / (1.0 - ab_k + 1e-8))[:, None] * np.ones((B, W))

        # Vectorised accumulation - no Python loop over BxW
        mse_np     = mse_day.cpu().numpy()          # (B, W)
        regime_np  = regime_bw_np                   # (B, W) int
        for r in range(n_regimes):
            mask_r = (regime_np == r)
            regime_mse[r] += float(mse_np[mask_r].sum())
            regime_n[r]   += int(mask_r.sum())

        snr_flat  = snr_np.ravel()
        mse_flat  = mse_np.ravel()
        lo_mask   = snr_flat < snr_lo
        hi_mask   = snr_flat > snr_hi
        mid_mask  = ~lo_mask & ~hi_mask
        snr_mse["low"]  += float(mse_flat[lo_mask].sum());  snr_n["low"]  += int(lo_mask.sum())
        snr_mse["mid"]  += float(mse_flat[mid_mask].sum()); snr_n["mid"]  += int(mid_mask.sum())
        snr_mse["high"] += float(mse_flat[hi_mask].sum());  snr_n["high"] += int(hi_mask.sum())

        del mse_day, v_pred

    out: Dict[str, float] = {}
    for i, nm in enumerate(regime_names):
        out[f"loss_{nm}"]   = regime_mse[i] / max(regime_n[i], 1)
        out[f"loss_n_{nm}"] = float(regime_n[i])
    for key in ("low", "mid", "high"):
        out[f"loss_snr_{key}"] = snr_mse[key] / max(snr_n[key], 1)

    tau_arr = np.array(all_tau_means) if all_tau_means else np.array([0.0])
    out["tau_mean"] = float(tau_arr.mean())
    out["tau_std"]  = float(tau_arr.std())

    # Per-class tau means - used for logging and separation diagnostics
    for i, nm in enumerate(regime_names):
        arr = np.array(tau_by_class[i]) if tau_by_class[i] else np.array([0.0])
        out[f"tau_mean_{nm}"] = float(arr.mean())

    # tau_separation = overcast_mean - clear_mean (backward compat key)
    tc_arr  = np.array(tau_by_class[0])                 if tau_by_class[0]                 else np.array([0.0])
    tcl_arr = np.array(tau_by_class[n_regimes - 1])     if tau_by_class[n_regimes - 1]     else np.array([0.0])
    out["tau_clear_mean"]  = float(tc_arr.mean())
    out["tau_cloudy_mean"] = float(tcl_arr.mean())
    out["tau_separation"]  = out["tau_cloudy_mean"] - out["tau_clear_mean"]
    return out


@torch.no_grad()
def _fast_generation_diagnostics(
    model: SolarDenoiser,
    vae: SolarVAE,
    val_loader: DataLoader,
    schedule: TauNoiseSchedule,
    latent_transform: LatentTauTransform,
    device: torch.device,
    cfg: Dict,
    n_batches: int = 8,
) -> Dict:
    #Fast generation-quality diagnostics - runs in ~10s instead of 25 minutes.
    # 
    # Instead of running full reverse diffusion and decoding (which dominated the
    # old transition-matrix diagnostic), this function:
    # 
    # 1. Latent diversity score  - denoises from k=200 (partial noise) to k=0 for
    #    n_batches of val data. Measures per-dim variance of generated tau vs real tau.
    #    Ratio < 0.4 signals mode collapse toward clear sky.
    # 
    # 2. K* marginal coverage    - decodes generated tau and checks fraction of days
    #    with mean K* in three bins: dark (<0.3 overcast), mid (0.3-0.7 mixed),
    #    bright (>0.7 clear). Compares against observed fractions.
    # 
    # 3. Intra-window K* variance - variance of per-day mean K* within each generated
    #    window.  A collapsing model produces near-constant K* across days (var~=0).
    #    Real weather has var ~= 0.03-0.10.
    # 
    # These three numbers together tell you:
    #   - diversity_ratio    : has the model learned regime spread?
    #   - coverage_dark_gap  : is overcast being generated?
    #   - window_var_ratio   : is there day-to-day variation within sequences?
    #
    dc        = cfg["diffusion"]
    K         = dc["num_steps"]
    # K* marginal coverage bin boundaries - from config, same scale as K* values.
    # Below _bin_lo -> overcast-like; above _bin_hi -> clear-like.
    _bin_lo   = float(cfg["diagnostics"].get("gen_coverage_bin_lo", 0.40))
    _bin_hi   = float(cfg["diagnostics"].get("gen_coverage_bin_hi", 0.70))

    # Partial reverse diffusion: k=800 -> k=0 using 20 DDPM steps.
    # Starting from 80% noise tests whether the model actually generates diverse
    # regimes rather than just reconstructing from low noise.  The previous
    # k_start=200 (20% noise) left regime structure intact in tau_k - any model
    # that minimally fits would pass diversity_ratio > 0.6, making the diagnostic
    # misleadingly optimistic even for a mode-collapsed model.
    k_start   = max(1, int(K * 0.80))
    n_steps   = 20
    step_list = sorted(
        list(range(k_start, 0, -(k_start // n_steps)))[:n_steps],
        reverse=True,
    )

    import math as _math_gen_diag
    _tau_max_diag = float(-_math_gen_diag.log(float(cfg["vae"].get("z_norm_clamp_lo", 0.01))))

    # Get standardised clamp bounds from the fitted transform.
    # After affine standardisation, tau is no longer in [0, tau_max_raw] so we
    # must use the transform's own bounds rather than the raw tau_max value.
    _tau_min_diag, _tau_max_diag_std = latent_transform.tau_clamp_bounds()

    model.eval()
    vae.eval()

    # Collect real and generated tau, plus decoded K* per day
    real_tau_vars:  List[float] = []   # per-dim variance of real tau0
    gen_tau_vars:   List[float] = []   # per-dim variance of generated tau
    real_mk_days:   List[float] = []   # observed mean K* per day
    gen_mk_days:    List[float] = []   # generated mean K* per day
    real_win_vars:  List[float] = []   # per-window variance of daily mean K*
    gen_win_vars:   List[float] = []   # per-window variance of daily mean K* (generated)

    n_done = 0
    for batch in val_loader:
        if n_done >= n_batches:
            break
        B, W, T_max = batch["k_star"].shape
        _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7

        k_bw  = batch["k_star"].to(device).reshape(B * W, T_max)
        ph_bw = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
        ma_bw = batch["valid_mask"].to(device).reshape(B * W, T_max)
        obs_bw = _get_obs_phys(batch, device, B, W, T_max)

        z, _  = vae.encode(k_bw, ph_bw, ma_bw, obs_bw)
        z      = z.reshape(B, W, -1)
        tau0   = latent_transform.to_tau_space(z)

        # Real tau per-dim variance across (B*W) samples
        tau0_flat = tau0.reshape(B * W, -1)   # (B*W, d_z)
        real_tau_vars.append(tau0_flat.var(dim=0).mean().item())

        # Real K* per day
        mk_real = _day_mean_k_tensor(k_bw, ma_bw).cpu().numpy()   # (B*W,)
        for mk in mk_real:
            real_mk_days.append(float(mk))
            
        regime_ids_diag = batch["regime"].to(device)   # (B, W) int64

        # A: track exact overcast/clear fractions using GMM regime labels
        real_regimes_batch = regime_ids_diag.reshape(B * W).cpu().numpy()
        _n_regimes_diag    = int(cfg["diffusion"].get("n_regimes", 4))
        _real_overcast_n   = int((real_regimes_batch == _n_regimes_diag - 1).sum())
        _real_clear_n      = int((real_regimes_batch == 0).sum())
        _real_total_n      = len(real_regimes_batch)

        # Real per-window variance
        mk_real_win = mk_real.reshape(B, W)
        for b in range(B):
            real_win_vars.append(float(np.var(mk_real_win[b])))

        # Partial reverse diffusion from k_start
        day_feat   = batch["day_features"].to(device)
        location   = batch["location"].to(device)
        intra_phys = batch["intraday_phys"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        drop_mask  = torch.zeros(B, device=device, dtype=torch.bool)

        # H10: start reverse diffusion from PURE Gaussian noise, not q_sample(tau0).
        # q_sample mixes real tau0 into the start: tau_k = sqrt(ab)*tau0 + sqrt(1-ab)*eps.
        # At k_start=800, ~44% of the real signal leaks through, making even a collapsed
        # model appear diverse because it partly reconstructs real data. Pure noise tests
        # true generation ability - the model receives no real-data signal at all.
        # tau0 is still used above for real_tau_vars (real distribution reference).
        tau_k = torch.randn_like(tau0)  # pure N(0,I) start - true generation condition
        # Scale to match the noise schedule variance at k_start:
        # at step k_start, E[tau_k^2] = 1 (unit variance pure noise matches N(0,I)).
        # The schedule already normalises sigma^2 to 1 at full noise, so randn is correct.


        _guidance = float(dc.get("cfg_guidance_scale", 1.0))
        for k in step_list:
            k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
            if _guidance > 1.0:
                v_pred = model.forward_cfg(
                    tau_k, k_tensor, day_feat, location,
                    intra_phys, valid_mask,
                    guidance_scale=_guidance,
                    regime_ids=regime_ids_diag,
                )
            else:
                v_pred = model(tau_k, k_tensor, day_feat, location,
                               intra_phys, valid_mask, drop_mask=drop_mask,
                               regime_ids=regime_ids_diag)
            # LEAK 3: explicitly delete old tau_k BEFORE overwriting.
            # Python only frees the old tensor when the reference is gone.
            # Without del, the old tau_k lives until the next p_sample call,
            # meaning TWO tau_k tensors exist simultaneously at peak - doubles
            # the memory footprint of the reverse loop.
            tau_k_old = tau_k
            tau_k = schedule.p_sample(tau_k, k, v_pred,
                                      tau_max=_tau_max_diag_std,
                                      tau_min=_tau_min_diag)
            del tau_k_old, v_pred  # free immediately - tight loop

        # Generated tau per-dim variance
        tau_gen_flat = tau_k.reshape(B * W, -1)
        gen_tau_vars.append(tau_gen_flat.var(dim=0).mean().item())

        # Decode generated tau -> K* per day (batched by T_sun group for speed).
        # This is diagnostic-only - the sequential loop was decoding B*W days
        # one at a time, each with a separate GPU kernel launch. On a 4GB GPU
        # with B=48, W=14 this was 672 sequential decode calls per diagnostic batch.
        _d_z_flat = int(cfg["vae"]["latent_dim"])
        _d_z_var  = int(cfg["vae"].get("z_var_dim", 0))
        tau_k_flat = tau_k[..., :_d_z_flat]
        z_flat_gen = latent_transform.from_tau_space(tau_k_flat)
        if _d_z_var > 0:
            z_var_zeros = torch.zeros(
                *tau_k.shape[:-1], _d_z_var, device=tau_k.device
            )
            z_gen = torch.cat([z_flat_gen, z_var_zeros], dim=-1)
        else:
            z_gen = z_flat_gen

        # Group days by T_sun so we can decode each group in one call
        t_sun_diag = valid_mask.reshape(B * W, -1).sum(dim=1).long()  # (B*W,)
        z_gen_flat = z_gen.reshape(B * W, -1)
        ph_flat    = intra_phys.reshape(B * W, intra_phys.shape[2], intra_phys.shape[3])

        from collections import defaultdict as _dd
        groups_diag = _dd(list)
        for idx, t in enumerate(t_sun_diag.cpu().tolist()):
            groups_diag[int(t)].append(idx)

        mk_gen_all = [float("nan")] * (B * W)
        for T_sun_val, idx_list in groups_diag.items():
            if T_sun_val == 0:
                continue
            z_b   = z_gen_flat[idx_list]
            ph_b  = torch.stack([ph_flat[i, :T_sun_val] for i in idx_list])
            k_hat = vae.decode(z_b, ph_b, T_sun_val)
            for j, orig_idx in enumerate(idx_list):
                mk_gen_all[orig_idx] = float(k_hat[j].mean().item())
            del k_hat

        for b in range(B):
            mk_gen_window = []
            for d in range(W):
                mk = mk_gen_all[b * W + d]
                if not (mk != mk):  # not nan
                    gen_mk_days.append(mk)
                mk_gen_window.append(mk)
            valid_mks = [v for v in mk_gen_window if not (v != v)]
            if valid_mks:
                gen_win_vars.append(float(np.var(valid_mks)))

        n_done += 1

    # -- Compute summary statistics --------------------------------------------
    real_var = float(np.mean(real_tau_vars)) if real_tau_vars else 1.0
    gen_var  = float(np.mean(gen_tau_vars))  if gen_tau_vars  else 0.0
    diversity_ratio = gen_var / max(real_var, 1e-8)

    def _bin_frac(mks, lo, hi):
        if not mks:
            return 0.0
        arr = np.array(mks)
        return float(np.mean((arr >= lo) & (arr < hi)))

    # A: K*-bin-based fractions are wrong because sunlit mean K* of
    # overcast days can be >0.40 (partial cloud cover). The logs showed
    # gen_real_dark=0 in EVERY diagnostic epoch, making the coverage check blind.
    # Also compute regime-label-based fractions (exact, no threshold needed).
    real_dark = _bin_frac(real_mk_days, 0.0,     _bin_lo)
    real_mid  = _bin_frac(real_mk_days, _bin_lo, _bin_hi)
    real_brt  = _bin_frac(real_mk_days, _bin_hi,  2.0)
    gen_dark  = _bin_frac(gen_mk_days,  0.0,     _bin_lo)
    gen_mid   = _bin_frac(gen_mk_days,  _bin_lo, _bin_hi)
    gen_brt   = _bin_frac(gen_mk_days,  _bin_hi,  2.0)

    real_wvar = float(np.mean(real_win_vars)) if real_win_vars else 1.0
    gen_wvar  = float(np.mean(gen_win_vars))  if gen_win_vars  else 0.0
    window_var_ratio = gen_wvar / max(real_wvar, 1e-8)

    return {
        "diversity_ratio":  diversity_ratio,   # >0.6 healthy, <0.3 collapsed
        "gen_var":          gen_var,
        "real_var":         real_var,
        "gen_dark":         gen_dark,          # fraction overcast days generated
        "gen_mid":          gen_mid,
        "gen_brt":          gen_brt,
        "real_dark":        real_dark,
        "real_mid":         real_mid,
        "real_brt":         real_brt,
        "coverage_dark_gap": real_dark - gen_dark,   # positive = model under-generates overcast
        "coverage_brt_gap":  gen_brt  - real_brt,    # positive = model over-generates clear
        "window_var_ratio":  window_var_ratio, # >0.5 healthy, <0.2 sequences are monotone
        "gen_wvar":          gen_wvar,
        "real_wvar":         real_wvar,
    }


def _log_gen_diagnostics(gd: Dict, cfg: Dict) -> None:
    #Log fast generation diagnostics.
    diversity_err  = float(cfg["diagnostics"].get("diversity_ratio_err",  0.3))
    diversity_warn = float(cfg["diagnostics"].get("diversity_ratio_warn", 0.6))
    dark_gap_warn  = float(cfg["diagnostics"].get("coverage_dark_gap_warn", 0.10))
    brt_gap_warn   = float(cfg["diagnostics"].get("coverage_brt_gap_warn",  0.10))
    wvar_warn      = float(cfg["diagnostics"].get("window_var_ratio_warn",  0.2))

    log.info(
        "  [GEN DIAG]  diversity=%.3f  window_var=%.3f  "
        "| dark: real=%.1f%%  gen=%.1f%%  (gap=%.1f%%)  "
        "| clear: real=%.1f%%  gen=%.1f%%  (gap=%.1f%%)",
        gd["diversity_ratio"], gd["window_var_ratio"],
        100 * gd["real_dark"], 100 * gd["gen_dark"],  100 * gd["coverage_dark_gap"],
        100 * gd["real_brt"],  100 * gd["gen_brt"],   -100 * gd["coverage_brt_gap"],
    )
    if gd["diversity_ratio"] < diversity_err:
        log.warning(
            "  [!!] diversity_ratio=%.3f < %.1f - mode-collapsing. "
            "Raise cfg_guidance_scale, check cfg_dropout_prob >= 0.15.",
            gd["diversity_ratio"], diversity_err,
        )
    elif gd["diversity_ratio"] < diversity_warn:
        log.warning("  [!] diversity_ratio=%.3f < %.1f - partial mode collapse.", gd["diversity_ratio"], diversity_warn)
    if gd["coverage_dark_gap"] > dark_gap_warn:
        log.warning(
            "  [!] Under-generating overcast: gap=%.1f%% - raise cfg_guidance_scale.",
            100 * gd["coverage_dark_gap"],
        )
    if gd["coverage_brt_gap"] > brt_gap_warn:
        log.warning("  [!] Over-generating clear sky: excess=%.1f%%.", 100 * gd["coverage_brt_gap"])
    if gd["window_var_ratio"] < wvar_warn:
        log.warning(
            "  [!] window_var_ratio=%.3f < %.1f - no day-to-day regime variation in sequences.",
            gd["window_var_ratio"], wvar_warn,
        )


def _log_diff_diagnostics(diag: Dict[str, float], cfg: Dict) -> None:
    n_regimes    = int(cfg["diffusion"].get("n_regimes", 4))
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:n_regimes]
    snr_lo       = float(cfg["diagnostics"].get("snr_low_boundary",  1.0))
    snr_hi       = float(cfg["diagnostics"].get("snr_high_boundary", 10.0))
    oc_loss_warn = float(cfg["diagnostics"].get("overcast_loss_ratio_warn", 4.0))
    snr_warn     = float(cfg["diagnostics"].get("snr_low_high_ratio_warn",  3.0))
    tau_sep_min  = float(cfg["diagnostics"]["tau_separation_min_to_proceed"])

    oc_nm = regime_names[-1]
    log.info(
        "  [VAL REGIME LOSS]  %s",
        "  ".join(f"{nm}=%.5f (N=%.0f)" % (diag.get(f"loss_{nm}", 0), diag.get(f"loss_n_{nm}", 0))
                 for nm in regime_names),
    )
    log.info(
        "  [SNR LOSS]         low(SNR<%.1f)=%.5f  mid=%.5f  high(SNR>%.1f)=%.5f",
        snr_lo, diag["loss_snr_low"], diag["loss_snr_mid"], snr_hi, diag["loss_snr_high"],
    )
    log.info(
        "  [TAU SPACE]        mean=%.3f  std=%.3f  | %s  sep(oc-clear)=%.3f",
        diag["tau_mean"], diag["tau_std"],
        "  ".join(
            "tau_%s=%.3f" % (nm, diag.get(f"tau_mean_{nm}", float("nan")))
            for nm in regime_names
        ),
        diag["tau_separation"],
    )
    lc  = diag.get("loss_clear", 0.0)
    lo  = diag.get(f"loss_{oc_nm}", 0.0)
    n_oc= diag.get(f"loss_n_{oc_nm}", 0)
    sl  = diag["loss_snr_low"]
    sh  = diag["loss_snr_high"]
    sep = diag["tau_separation"]
    if n_oc == 0:
        log.warning("  [!] No %s days in val set (N=0) - add %s-rich stations.", oc_nm, oc_nm)
    elif lc > 0 and lo > oc_loss_warn * lc:
        log.warning(
            "  [!] val_loss_%s %.1fx > val_loss_clear - diffusion biased toward clear. "
            "Raise regime_loss_weights_diff for %s.", oc_nm, lo / max(lc, 1e-8), oc_nm,
        )
    if sh > 0 and sl > snr_warn * sh:
        log.warning(
            "  [!] loss_snr_low %.1fx > loss_snr_high - model struggles at high noise. "
            "Raise regime_loss_weights_diff or add stations.", sl / max(sh, 1e-8),
        )
    if sep < tau_sep_min:
        log.warning(
            "  [!] tau_separation=%.3f < %.2f - rebuild latent cache and re-fit LatentTauTransform.",
            sep, tau_sep_min,
        )


def _diff_action_guide(
    best_diag: Dict[str, float],
    last_gen_diag: Optional[Dict],
    best_train_metrics: Dict[str, float],
    cfg: Dict,
) -> None:
    dc           = cfg["diffusion"]
    regime_names = cfg["data"]["regime"].get(
        "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
    )[:dc.get("n_regimes", 4)]
    tau_sep_min     = float(cfg["diagnostics"]["tau_separation_min_to_proceed"])
    cloudy_ratio_warn = float(cfg["diagnostics"].get("cloudy_clear_ratio_warn", 4.0))
    oc_loss_ratio_warn= float(cfg["diagnostics"].get("overcast_loss_ratio_warn", 4.0))
    snr_ratio_warn  = float(cfg["diagnostics"].get("snr_low_high_ratio_warn", 3.0))
    diversity_warn  = float(cfg["diagnostics"].get("diversity_ratio_warn", 0.4))
    dark_gap_warn   = float(cfg["diagnostics"].get("coverage_dark_gap_warn", 0.10))

    log.info("=" * 72)
    log.info("STAGE 2 COMPLETE -- ACTION GUIDE")
    log.info("=" * 72)

    lc    = best_train_metrics.get("loss_clear",  0.0)
    lcl   = best_train_metrics.get("loss_cloudy", 0.0)
    ratio = lcl / max(lc, 1e-8)
    log.info("  Train cloudy/clear loss ratio : %.2f  (target 1-2)", ratio)
    log.info("  Val regime loss  : %s",
             "  ".join(f"{nm}=%.5f" % best_diag.get(f"loss_{nm}", 0) for nm in regime_names))
    log.info("  Val SNR loss     : low=%.5f  mid=%.5f  high=%.5f",
             best_diag.get("loss_snr_low",  0),
             best_diag.get("loss_snr_mid",  0),
             best_diag.get("loss_snr_high", 0))
    log.info("  Tau separation   : %.3f  (target > %.2f)", best_diag.get("tau_separation", 0), tau_sep_min)

    if last_gen_diag is not None:
        log.info("  Gen diversity    : %.3f  (target > %.1f)", last_gen_diag.get("diversity_ratio", 0), diversity_warn)
        log.info("  Window var ratio : %.3f", last_gen_diag.get("window_var_ratio", 0))
        log.info("  Coverage dark    : real=%.1f%%  gen=%.1f%%  (gap=%.1f%%)",
                 100 * last_gen_diag.get("real_dark", 0),
                 100 * last_gen_diag.get("gen_dark",  0),
                 100 * last_gen_diag.get("coverage_dark_gap", 0))
        log.info("  Coverage clear   : real=%.1f%%  gen=%.1f%%  (excess=%.1f%%)",
                 100 * last_gen_diag.get("real_brt",  0),
                 100 * last_gen_diag.get("gen_brt",   0),
                 100 * last_gen_diag.get("coverage_brt_gap", 0))

    issues = []
    max_diff_w = max(dc["regime_loss_weights_diff"])
    oc_nm = regime_names[-1]
    lo  = best_diag.get(f"loss_{oc_nm}", 0.0)
    lcc = best_diag.get("loss_clear",    0.0)

    if ratio > cloudy_ratio_warn:
        issues.append(
            f"  RAISE regime_loss_weights_diff (max currently {max_diff_w:.1f}): "
            f"cloudy/clear ratio={ratio:.2f} >> 2."
        )
    if lcc > 0 and lo > oc_loss_ratio_warn * lcc:
        issues.append(
            f"  RAISE regime_loss_weights_diff for {oc_nm}: "
            f"val_loss_{oc_nm} is {lo/max(lcc,1e-8):.1f}x val_loss_clear."
        )
    sl = best_diag.get("loss_snr_low",  0.0)
    sh = best_diag.get("loss_snr_high", 0.0)
    if sh > 0 and sl > snr_ratio_warn * sh:
        issues.append(
            f"  RAISE regime_loss_weights_diff AND/OR add more stations: "
            f"loss_snr_low is {sl/max(sh,1e-8):.1f}x loss_snr_high."
        )
    if best_diag.get("tau_separation", 1.0) < tau_sep_min:
        issues.append(
            f"  Rebuild latent cache and re-fit LatentTauTransform: "
            f"tau_separation < {tau_sep_min:.2f}."
        )
    if last_gen_diag is not None:
        if last_gen_diag.get("diversity_ratio", 1.0) < diversity_warn:
            issues.append(
                f"  Mode collapse (diversity={last_gen_diag['diversity_ratio']:.3f} < {diversity_warn:.1f}). "
                f"Check cfg_guidance_scale, cfg_dropout_prob, window_stride."
            )
        if last_gen_diag.get("coverage_dark_gap", 0.0) > dark_gap_warn:
            issues.append(
                f"  Under-generating {oc_nm} (gap={last_gen_diag['coverage_dark_gap']:.2f}). "
                f"Raise cfg_guidance_scale or regime_loss_weights_diff for {oc_nm}."
            )

    if issues:
        log.warning("  RECOMMENDED CHANGES:")
        for msg in issues:
            log.warning(msg)
    else:
        log.info("  All metrics look healthy.")
    log.info("=" * 72)


# =============================================================================
# Stage 1 - AE epoch runner
# =============================================================================

def _run_vae_epoch(
    model: SolarVAE,
    loader: DataLoader,
    optim,
    device: torch.device,
    cfg: Dict,
    grad_clip: float,
    log_every: int,
    train: bool,
    epoch: int = 0,
    spectral_power_mean: Optional[torch.Tensor] = None,  # (n_freq,) whitening weights
) -> Dict[str, float]:
    #Run one AE epoch.
    # 
    # Returns recon / spectral / sep / total / grad_norm / sep_fired_pct.
    # 
    # Every step: forward -> ae_loss -> regime_separation_loss -> backward.
    # sep_loss == 0.0 once clear-cloudy centroid distance >= regime_sep_margin;
    # that is correct and healthy - the loss only fires when separation is insufficient.
    #
    # Spectral weight warmup: ramp from 0.0 to cfg value over first 10 epochs
    # for the TRAINING loss only.  The val spectral loss always uses the full
    # target weight so it remains comparable across all epochs (a warmed-up val
    # weight would produce artificially low spectral values in early epochs,
    # making early-stopping on spectral loss unreliable).
    spectral_w_target = cfg["vae"]["spectral_weight"]
    spectral_warmup_epochs = cfg["vae"].get("spectral_warmup_epochs", 10)
    if train and epoch < spectral_warmup_epochs:
        # Use same linear ramp as LinearLR: factor = (epoch+1)/total_iters
        # reaches 1.0 exactly at epoch == spectral_warmup_epochs (same as LR).
        spectral_w = spectral_w_target * min((epoch + 1) / max(spectral_warmup_epochs, 1), 1.0)
    else:
        spectral_w = spectral_w_target
    total_recon   = 0.0
    total_spec    = 0.0
    total_sep     = 0.0
    total_var     = 0.0
    total_loss    = 0.0
    total_gnorm   = 0.0
    n_fired       = 0
    n_batches     = 0
    total_attn_entropy = 0.0   # mean encoder attention entropy - tracks focus sharpness
    n_attn_samples     = 0     # steps where attn_entropy was actually measured

    # -- numerical-health accumulators (subtle problem detection) --------------
    total_dead_ratio  = 0.0
    total_khat_max    = 0.0
    total_khat_min    = 0.0
    total_z_std       = 0.0
    total_sep_dist    = 0.0
    n_sep_dist_obs    = 0
    n_nan_loss        = 0
    n_loss_spike      = 0
    running_loss_mean = 0.0
    _loss_spike_mult  = float(cfg["training"]["vae"].get("loss_spike_multiplier", 10.0))
    _loss_ema_alpha   = float(cfg["training"]["vae"].get("loss_ema_alpha", 0.05))
    # EMA of per-dim z_std across the epoch, used for the variance penalty.
    # A single-batch std is too noisy - a batch that happens to have high variance
    # silences the penalty even if the global distribution is collapsed.
    # EMA with the same alpha as loss smoothing (~20-50 batch memory) gives a stable signal.
    _z_std_ema: Optional[torch.Tensor] = None
    _z_std_ema_alpha = _loss_ema_alpha   # reuse same smoothing factor

    # P8: no threshold variables needed - regime comes from batch["regime"]

    for batch in loader:
        B, W, T_max = batch["k_star"].shape
        _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7
        k_star = batch["k_star"].to(device).reshape(B * W, T_max)
        phys   = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
        mask   = batch["valid_mask"].to(device).reshape(B * W, T_max)
        obs    = _get_obs_phys(batch, device, B, W, T_max)

        # -- dead-mask check: fraction of timesteps that are masked (nighttime)
        dead_ratio = 1.0 - mask.float().mean().item()
        total_dead_ratio += dead_ratio

        k_hat, z = model(k_star, phys, mask, obs)

        # H8: Update EMA of per-dim z_std using only z_flat dims.
        # model() returns z_full = cat(z_flat, z_var). z_var dims have a
        # different (narrower) variance and inflate/deflate the z_std signal
        # that drives the variance penalty and controller. Slice to z_flat only.
        _d_z_flat_run = int(cfg["vae"]["latent_dim"])  # z_flat dim count
        with torch.no_grad():
            batch_z_std = z[:, :_d_z_flat_run].detach().std(dim=0)   # (latent_dim,)
            if _z_std_ema is None:
                _z_std_ema = batch_z_std.clone()
            else:
                _z_std_ema.mul_(1.0 - _z_std_ema_alpha).add_(batch_z_std, alpha=_z_std_ema_alpha)

        attn_entropy_val = float("nan")
        if train and log_every > 0 and n_batches % log_every == 0:
            with torch.no_grad():
                _, attn_entropy_val_t = model.encode_with_attn_entropy(k_star, phys, mask, obs)
                attn_entropy_val = attn_entropy_val_t.item()
        if not (attn_entropy_val != attn_entropy_val):
            total_attn_entropy += attn_entropy_val
            n_attn_samples     += 1

        # Per-sample loss weights indexed by GMM regime label.
        # regime_loss_weights in config lists one weight per class in order
        # [clear, mixed_clear, mixed_overcast, overcast].  All other terms
        # (spectral, ramp, curvature) receive the same weight, so the harder
        # regimes get proportionally more gradient on every loss component.
        regime_weights_cfg = cfg["vae"]["regime_loss_weights"]
        regime_ids_bw = _regime_from_batch(batch, device)
        sample_w = torch.ones(regime_ids_bw.shape[0], device=device)
        for cls_idx, w in enumerate(regime_weights_cfg):
            sample_w[regime_ids_bw == cls_idx] = float(w)

        ae_total, recon = ae_loss(
            k_hat, k_star, mask, spectral_w,
            ramp_threshold=float(cfg["vae"]["ramp_significance_threshold"]),
            phase_weight=float(cfg["vae"]["phase_weight"]),
            curvature_weight=float(cfg["vae"]["curvature_weight"]),
            phase_amp_threshold=float(cfg["vae"]["phase_amp_threshold"]),
            ramp_offset_steps=int(cfg["vae"]["ramp_offset_steps"]),
            sample_weight=sample_w,
            spectral_power_mean=spectral_power_mean,
        )

        # Separation loss uses GMM regime ids (no thresholds).
        # Guard: ONLY call on training passes.  The EMA inside regime_separation_loss
        # is module-level Python state - torch.no_grad() does NOT suppress Python dict
        # updates, so the val pass would corrupt the training margin target (Case 12).
        if train:
            sep_loss, fired = regime_separation_loss(z, cfg, regime_ids=regime_ids_bw, tau0=None, stage="vae")
        else:
            sep_loss = z.new_zeros(1).squeeze()
            fired    = False

        var_loss = latent_variance_penalty(
            z,
            z_std_target=float(cfg["vae"]["z_std_target"]),
            variance_weight=float(cfg["vae"]["z_var_weight"]),
            smoothed_z_std=_z_std_ema,
            z_std_upper_target=float(cfg["vae"]["z_std_upper_target"])
                if cfg["vae"].get("z_std_upper_target") is not None else None,
            variance_weight_upper=float(cfg["vae"].get("z_var_weight_upper", 0.0)),
        )

        loss = ae_total + sep_loss + var_loss

        spectral_val = (ae_total.item() - recon.item()) / max(spectral_w, 1e-8)

        # -- numerical health checks -------------------------------------------
        loss_val = loss.item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            n_nan_loss += 1
            log.warning(
                "  [!] NaN/Inf loss at step=%d  recon=%.4f  spec=%.4f  sep=%.5f  "
                "k_hat range=[%.3f, %.3f]  z range=[%.3f, %.3f]",
                n_batches, recon.item(), spectral_val, sep_loss.item(),
                k_hat.min().item(), k_hat.max().item(),
                z.min().item(), z.max().item(),
            )
        if running_loss_mean > 0 and loss_val > _loss_spike_mult * running_loss_mean:
            n_loss_spike += 1
            log.warning(
                "  [!] Loss spike at step=%d: %.4f vs running mean %.4f (%.1fx) -- "
                "skipping backward pass to protect gradients. "
                "Root cause: K*>1.0 in raw data - check k_max_clip in cfg.",
                n_batches, loss_val, running_loss_mean, loss_val / running_loss_mean,
            )
            # Do NOT backpropagate a spike - the gradient would be 10-25x normal
            # magnitude and corrupt FiLM layers even with grad_clip=1.0.
            # Update running_mean with a clamped value so future threshold stays sane.
            running_loss_mean = (1.0 - _loss_ema_alpha) * running_loss_mean + _loss_ema_alpha * min(loss_val, 2.0 * running_loss_mean)
            n_batches += 1
            continue
        running_loss_mean = (1.0 - _loss_ema_alpha) * running_loss_mean + _loss_ema_alpha * loss_val if running_loss_mean > 0 else loss_val

        with torch.no_grad():
            total_khat_max += k_hat.max().item()
            total_khat_min += k_hat.min().item()
            # H8: track z_flat std only (z_var dims excluded)
            total_z_std    += z[:, :_d_z_flat_run].std().item()
            # P8: batch clear-overcast centroid dist using GMM labels
            # overcast is always the last class (n_regimes - 1)
            n_regimes_local = int(cfg["diffusion"].get("n_regimes", 4))
            is_clear_b    = (regime_ids_bw == 0)
            is_overcast_b = (regime_ids_bw == n_regimes_local - 1)
            if is_clear_b.sum() >= 2 and is_overcast_b.sum() >= 2:
                mu_c  = z[is_clear_b].mean(dim=0)
                mu_oc = z[is_overcast_b].mean(dim=0)
                total_sep_dist += (mu_c - mu_oc).norm(p=2).item()
                n_sep_dist_obs += 1

        if train:
            optim.zero_grad()
            loss.backward()
            gnorm = _compute_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            total_gnorm += gnorm

        total_recon += recon.item()
        total_spec  += spectral_val
        total_sep   += sep_loss.item()
        total_var   += var_loss.item()
        total_loss  += loss_val

        
        n_fired     += int(fired)
        n_batches   += 1

        if train and log_every > 0 and n_batches % log_every == 0:
            log.debug(
                "  AE step=%d  recon=%.4f  spectral=%.4f  sep=%.5f(fired=%s)  "
                "total=%.4f  gnorm=%.3f  z_std=%.3f  attn_H=%.3f  dead=%.2f  khat=[%.3f,%.3f]",
                n_batches, recon.item(), spectral_val,
                sep_loss.item(), "Y" if fired else "N",
                loss_val, total_gnorm / n_batches,
                z[:, :_d_z_flat_run].std().item(),  # z_flat std only (H8)
                total_attn_entropy / max(n_attn_samples, 1),
                dead_ratio,
                k_hat.min().item(), k_hat.max().item(),
            )
            
        # LEAK 6: free k_hat and z after each VAE batch.
        # k_hat is (B*W, T_max) and z is (B*W, d_z) - both GPU tensors.
        # Without explicit del, Python holds them until the next batch
        # overwrites the names, meaning two copies exist at peak.
        del k_hat, z, k_star, phys, mask
        if obs is not None:
            del obs

    nb = max(n_batches, 1)
    avg_sep_dist = total_sep_dist / max(n_sep_dist_obs, 1)
    return {
        "recon":            total_recon      / nb,
        "spectral":         total_spec       / nb,
        "sep":              total_sep        / nb,
        "total":            total_loss       / nb,
        "grad_norm":        total_gnorm      / nb,
        "sep_fired_pct":    100.0 * n_fired  / nb,
        "var_loss":         total_var        / nb,
        "attn_entropy":     total_attn_entropy / max(n_attn_samples, 1),
        # numerical health
        "dead_ratio":       total_dead_ratio / nb,
        "khat_max":         total_khat_max   / nb,
        "khat_min":         total_khat_min   / nb,
        # M7: log both the epoch-average z_std and the EMA used by the penalty.
        # "z_std" = mean of per-batch z_flat.std() - epoch-level average (noisy).
        # "z_std_ema" = the smoothed EMA tensor mean that the variance penalty saw.
        # The penalty fires based on z_std_ema, so z_std_ema is the authoritative signal.
        "z_std":            total_z_std      / nb,
        "z_std_ema":        float(_z_std_ema.mean().item()) if _z_std_ema is not None else float("nan"),
        "batch_sep_dist":   avg_sep_dist,
        "n_nan_loss":       float(n_nan_loss),
        "n_loss_spike":     float(n_loss_spike),
    }


def _log_per_layer_grads_from_loader(
    model: nn.Module,
    loader: DataLoader,
    optim,
    device: torch.device,
    cfg: Dict,
    epoch: int,
    stage: str,
) -> None:
    #Log per-layer gradient norms using a single diagnostic backward pass.
    # 
    # Uses full target spectral_weight (not warmed-up) so gradient magnitudes
    # are comparable across epochs.  Gradients are zeroed before AND after so
    # the optimizer state is left clean regardless of call site ordering.
    # This is called at most every grad_log_freq epochs (1 batch overhead).
    #
    spectral_w = cfg["vae"]["spectral_weight"]   # always use full weight for comparability
    model.train()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    B, W, T_max = batch["k_star"].shape
    _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7
    k_star = batch["k_star"].to(device).reshape(B * W, T_max)
    phys   = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
    mask   = batch["valid_mask"].to(device).reshape(B * W, T_max)
    obs    = _get_obs_phys(batch, device, B, W, T_max)
    k_hat, z = model(k_star, phys, mask, obs)
    ae_total, _ = ae_loss(k_hat, k_star, mask, spectral_w,
                          ramp_threshold=float(cfg["vae"]["ramp_significance_threshold"]),
                          phase_weight=float(cfg["vae"]["phase_weight"]),
                          curvature_weight=float(cfg["vae"]["curvature_weight"]),
                          phase_amp_threshold=float(cfg["vae"]["phase_amp_threshold"]),
                          ramp_offset_steps=int(cfg["vae"]["ramp_offset_steps"]),
                          spectral_power_mean=None)
    # P8: use GMM regime_ids from batch - matches training loop exactly
    regime_ids_bw = _regime_from_batch(batch, device)
    sep_l, _ = regime_separation_loss(z, cfg, regime_ids=regime_ids_bw, tau0=None, stage="vae")
    total_loss = ae_total + sep_l
    optim.zero_grad()          # clear any stale grads before backward
    total_loss.backward()
    _log_per_layer_grads(model, epoch + 1, stage)
    optim.zero_grad()          # discard diagnostic grads - do NOT step
    model.eval()


# =============================================================================
# Stage 1 - main training loop
# =============================================================================

def train_vae(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    resume: bool = False,
) -> SolarVAE:
    #Train the physics-conditioned deterministic Autoencoder.
    # Clear any stale separation-EMA state from a prior training run that may
    # have run in the same Python process (e.g. a hyperparameter sweep).
    # Must be called before the model is created so the EMA is fresh from epoch 0.
    from solar_diffusion.vae import reset_sep_ema
    reset_sep_ema("vae")

    device = get_device(cfg)
    tc     = cfg["training"]["vae"]

    model = SolarVAE(cfg).to(device)
    optim = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    # Bug fix: Stage 1 was missing LR warmup, causing full LR to hit the encoder
    # in epochs 0-2 while spectral_weight was only 10-20% of target (spectral warmup).
    # Large reconstruction gradients at full LR compressed z_std before any
    # ramp/spectral pressure could resist it - root cause of latent collapse.
    # Mirror Stage 2's warmup+cosine schedule. warmup_epochs defaults to 5 if
    # not present in cfg["training"]["vae"] (safe for existing configs).
    _vae_warmup_epochs = int(tc.get("warmup_epochs", 5))
    _vae_warmup_epochs = max(1, min(_vae_warmup_epochs, tc["epochs"] - 1))
    _warmup = LinearLR(optim, start_factor=tc.get("warmup_start_factor", 0.1),
                       end_factor=1.0, total_iters=_vae_warmup_epochs)
    _cosine = CosineAnnealingLR(optim,
                                T_max=max(tc["epochs"] - _vae_warmup_epochs, 1),
                                eta_min=tc["lr_min"])
    sched = SequentialLR(optim, schedulers=[_warmup, _cosine],
                         milestones=[_vae_warmup_epochs])

    best_val      = float("inf")
    best_path     = Path(cfg["paths"]["vae_checkpoint"])
    # Z_STD-AWARE CHECKPOINT: save a separate checkpoint that tracks the best
    # val_recon among epochs where z_std is healthy (>= z_std_target * 0.65).
    # val_recon alone is misleading - it keeps improving as the latent collapses
    # because the encoder maps everything to the dataset mean and the decoder
    # reconstructs that mean accurately.  The z_std-gated checkpoint is the
    # one that should be used for Stage 2, because it has a properly spread
    # latent space.  The unconstrained best_path is kept for diagnostic purposes.
    _z_std_target   = float(cfg["vae"].get("z_std_target", 0.55))
    _z_std_healthy  = _z_std_target * 0.65        # e.g. 0.55 * 0.65 = 0.358
    best_val_healthy = float("inf")
    best_path_healthy = best_path.parent / (best_path.stem + "_healthy" + best_path.suffix)
    # Periodic checkpoint path - overwritten every save_every_n_epochs epochs.
    _save_every      = int(cfg.get("logging", {}).get("save_every_n_epochs", 20))
    periodic_path_vae = best_path.parent / (best_path.stem + "_periodic" + best_path.suffix)
    patience      = tc["early_stopping_patience"]
    epochs_no_imp = 0
    start_epoch   = 0

    if resume and best_path.exists():
        ckpt = load_checkpoint(best_path, model, optim)
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val      = ckpt.get("metric", float("inf"))
        # L4: restore epochs_no_imp so patience is correct on resume.
        # Previously patience always restarted at 0, giving the resumed model
        # a full fresh patience window even if it had already consumed some.
        epochs_no_imp = ckpt.get("epochs_no_imp", 0)
        # H6 (verified correct): SequentialLR.__init__ calls step() once.
        # Loop runs (start_epoch - 1) additional steps -> total = start_epoch steps.
        # This is correct: after training epoch k, sched has been stepped k+1 times.
        # On resume at start_epoch=k+1 we need k+1 total = 1 (init) + k (loop).
        for _ in range(max(start_epoch - 1, 0)):
            sched.step()
        log.info(
            "Resuming VAE training from epoch %d  best_val=%.5f  epochs_no_imp=%d",
            start_epoch, best_val, epochs_no_imp,
        )
        # H3: Reset z_var EMA so stale normalisation state from the previous
        # run does not persist. The EMA will re-warm quickly (within 20 batches)
        # due to the fast-warmup logic added to _ZVarHead._normalise().
        if hasattr(model, "z_var_head") and hasattr(model.z_var_head, "force_reset_ema"):
            model.z_var_head.force_reset_ema()
            log.info("z_var EMA state reset for resumed run (H3).")
    log_every     = tc["log_every_n_steps"]
    grad_log_freq = tc["log_grad_every_n_epochs"]
    diag_freq     = cfg.get("diagnostics", {}).get("vae_diag_freq", 5)
    best_diag: Dict[str, float] = {}

    # JSON training log - one record per epoch, append-only for safe resume.
    vae_jsonlog = Path(cfg.get("paths", {}).get(
        "vae_training_log", "logs/vae_training_log.jsonl"))
    log.info("  JSON epoch log -> %s", vae_jsonlog)

    # -- Spectral power mean for FFT whitening ---------------------------------
    # Accumulated online during epoch 0 from training data.
    # Held fixed thereafter and passed to ae_loss() as whitening weights.
    # Shape: (n_freq,) = (T_max // 2 + 1,). Computed on CPU, moved to device.
    spectral_power_mean: Optional[torch.Tensor] = None

    log.info("=== Stage 1: AE training for %d epochs (patience=%d) ===",
             tc["epochs"], patience)
    log.info(
        "  lr=%.2e  lr_min=%.2e  grad_clip=%.1f  weight_decay=%.2e  "
        "spectral_w=%.3f  sep_w=%.3f  dyn_margin_alpha=%.2f  "
        "mixed_sep_w=%.3f  mixed_margin_scale=%.2f",
        tc["lr"], tc["lr_min"], tc["grad_clip"], tc["weight_decay"],
        cfg["vae"]["spectral_weight"],
        cfg["vae"].get("regime_sep_weight", 2.0),
        cfg["vae"].get("dynamic_margin_alpha", 0.70),
        cfg["vae"].get("mixed_sep_weight", 2.0),
        cfg["vae"].get("mixed_margin_scale", 0.50),
    )

    # -- Startup config sanity checks ------------------------------------------
    _k_max      = float(cfg["physics"]["k_max"])
    _k_max_clip = float(cfg["data"]["augmentation"].get("k_max_clip", _k_max))
    _phase_w    = float(cfg["vae"].get("phase_weight", 0.20))
    _spec_w     = float(cfg["vae"]["spectral_weight"])
    _k_max_clip_warn  = float(cfg["vae"].get("k_max_clip_warn_threshold", 1.50))
    _phase_w_warn_lo  = float(cfg["vae"].get("phase_weight_warn_lo", 0.15))
    _phase_w_warn_rec = float(cfg["vae"].get("phase_weight_warn_recommended", 0.20))
    _spec_w_warn_hi   = float(cfg["vae"].get("spectral_weight_warn_hi", 0.50))
    if _k_max_clip > _k_max_clip_warn:
        log.warning(
            "[STARTUP] k_max_clip=%.2f is above %.2f. Values this high likely "
            "include sensor artefacts beyond the cloud-enhancement regime "
            "(p99 95th-pct=1.699). Consider capping at %.2f.",
            _k_max_clip, _k_max_clip_warn, _k_max,
        )
    if _phase_w < _phase_w_warn_lo and _spec_w >= _spec_w_warn_hi:
        log.warning(
            "[STARTUP] phase_weight=%.2f < %.2f with spectral_weight=%.2f. "
            "Phase loss penalises ramp timing - the main mixed-day gap. "
            "Raise phase_weight to >= %.2f to close the mixed/clear recon ratio.",
            _phase_w, _phase_w_warn_lo, _spec_w, _phase_w_warn_rec,
        )
    _n_reg = int(cfg["diffusion"].get("n_regimes", 4))
    _vae_weights = cfg["vae"].get("regime_loss_weights", [])
    if len(_vae_weights) != _n_reg:
        raise ValueError(
            f"[STARTUP] vae.regime_loss_weights has {len(_vae_weights)} entries "
            f"but diffusion.n_regimes={_n_reg}. They must match exactly."
        )
    log.info("  Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # -- Adaptive training controller ------------------------------------------
    # Monitors gnorm, sep loss, and z_std each epoch and adjusts cfg in-place.
    # tc["grad_clip"] is re-read from cfg at the start of each epoch so any
    # adjustment takes effect on the very next call to _run_vae_epoch.
    _vae_controller = TrainingController(cfg, stage="vae")

    for epoch in range(start_epoch, tc["epochs"]):
        # -- Always compute FFT power spectrum at the start of training --------
        # Note 3 FIX: previously this block was gated on `epoch == start_epoch`,
        # which on a fresh run means epoch=0 (runs correctly) but on resume means
        # epoch=N>0 (still runs - but spectral_power_mean was None before the
        # loop, so it was silently None for the entire resumed run, disabling FFT
        # whitening and changing the spectral gradient direction mid-training).
        # Fix: always compute the spectrum at the very first iteration of the
        # loop, i.e. when spectral_power_mean is still None.  This fires once
        # per process start regardless of start_epoch.
        if spectral_power_mean is None:
            log.info("  Computing FFT power spectrum for spectral whitening (epoch %d pass)...", start_epoch)
            # Use a fixed FFT length = cfg inference_steps_per_day (the canonical day length).
            # This ensures every batch produces the same n_freq = n_fft//2+1 regardless
            # of the variable T_max padding in each batch.
            n_fft     = int(cfg["data"].get("inference_steps_per_day", 144))
            n_freq    = n_fft // 2 + 1
            power_acc = torch.zeros(n_freq, device=device)
            power_n   = 0
            with torch.no_grad():
                for _batch in train_loader:
                    _B, _W, _T = _batch["k_star"].shape
                    _k  = _batch["k_star"].to(device).reshape(_B * _W, _T)
                    _m  = _batch["valid_mask"].to(device).reshape(_B * _W, _T)
                    _n_valid = _m.sum(dim=1).long()   # (B*W,) sunlit lengths
                    # FFT per day on sunlit slice - matches ae_loss fix.
                    # Zeroing then FFT inflates high-freq power via the hard
                    # sunrise/sunset step, making whitening weights wrong.
                    for i in range(_B * _W):
                        L = int(_n_valid[i].item())
                        if L < 2:
                            continue
                        _fft = torch.fft.rfft(_k[i, :L], n=n_fft, norm="ortho")
                        power_acc += _fft.abs().pow(2)
                        power_n   += 1
            spectral_power_mean = power_acc / max(power_n, 1)     # (n_freq,) on device
            log.info(
                "  FFT power spectrum computed: n_fft=%d  n_freq=%d  "
                "mean_power=%.4f  max_power=%.4f  min_power=%.6f",
                n_fft, n_freq,
                spectral_power_mean.mean().item(),
                spectral_power_mean.max().item(),
                spectral_power_mean.min().item(),
            )

        model.train()
        train_m = _run_vae_epoch(
            model, train_loader, optim, device, cfg,
            tc["grad_clip"], log_every, train=True, epoch=epoch,
            spectral_power_mean=spectral_power_mean,
        )

        model.eval()
        with torch.no_grad():
            val_m = _run_vae_epoch(
                model, val_loader, None, device, cfg,
                tc["grad_clip"], log_every, train=False, epoch=epoch,
                spectral_power_mean=spectral_power_mean,
            )

        sched.step()
        param_norm = _compute_param_norm(model)

        _cur_z_std = float(train_m.get("z_std", 0.0))

        # Best overall (val_recon only) - kept for diagnostics.
        is_best = val_m["recon"] < best_val
        if is_best:
            best_val      = val_m["recon"]
            save_checkpoint(best_path, model, optim, epoch, val_m["recon"])

        # Z_STD-AWARE BEST: only update when the latent space is healthy.
        # This is the checkpoint to use for Stage 2 - do NOT use best_path
        # (unconstrained) because val_recon improves monotonically even as
        # z_std collapses, producing a useless latent space for the diffusion.
        _z_std_ok = _cur_z_std >= _z_std_healthy
        if _z_std_ok and val_m["recon"] < best_val_healthy:
            best_val_healthy = val_m["recon"]
            save_checkpoint(best_path_healthy, model, optim, epoch, val_m["recon"],
                            extra={"z_std": _cur_z_std})
            log.info("  [HEALTHY CKPT] z_std=%.3f >= %.3f - saved healthy checkpoint "                     "(val_recon=%.5f) -> %s",
                     _cur_z_std, _z_std_healthy, val_m["recon"], best_path_healthy)

        # Periodic checkpoint - overwritten every _save_every epochs.
        # Useful to recover from a run that diverged after the best checkpoint.
        if (epoch + 1) % _save_every == 0:
            save_checkpoint(periodic_path_vae, model, optim, epoch, val_m["recon"],
                            extra={"z_std": _cur_z_std, "periodic": True})
            log.info("  [PERIODIC CKPT] epoch=%d saved -> %s", epoch + 1, periodic_path_vae)

        # M3: Revised patience logic - distinguish improving vs stagnating.
        # Old logic: patience increments whenever z_std < healthy OR val not best.
        # Problem: a temporary z_std dip during recovery stops training prematurely.
        #
        # New logic:
        #   val_recon improved + z_std healthy -> reset patience (model fully healthy)
        #   val_recon improved + z_std unhealthy -> pause patience (hold; do not reset)
        #     Model is improving reconstruction; z_std collapse is temporary.
        #     Controller should raise z_var_weight to address it - stopping is wrong.
        #   val_recon NOT improved -> always increment patience
        if is_best:
            if _z_std_ok:
                # Fully healthy improvement: reset patience
                epochs_no_imp = 0
            else:
                # Reconstruction improving but z_std unhealthy: pause patience.
                # Do not reset (requires healthy z_std) but do not increment either.
                log.info(
                    "  [PATIENCE] val_recon improved but z_std=%.3f < healthy %.3f - "
                    "pausing patience at %d (not incrementing while recovering).",
                    _cur_z_std, _z_std_healthy, epochs_no_imp,
                )
        else:
            # No val_recon improvement -> increment patience regardless of z_std
            epochs_no_imp += 1

        log.info(
            "Epoch %3d/%d | "
            "train [recon=%.5f  spec=%.5f  sep=%.5f(fired=%.0f%%)  var=%.5f  total=%.5f  gnorm=%.3f] | "
            "val [recon=%.5f  spec=%.5f  total=%.5f] | "
            "lr=%.2e  pnorm=%.3f  no_imp=%d%s",
            epoch + 1, tc["epochs"],
            train_m["recon"], train_m["spectral"],
            train_m["sep"], train_m["sep_fired_pct"],
            train_m.get("var_loss", 0.0),
            train_m["total"], train_m["grad_norm"],
            val_m["recon"],   val_m["spectral"],   val_m["total"],
            optim.param_groups[0]["lr"], param_norm, epochs_no_imp,
            "  BEST" if is_best else "",
        )
        log.info(
            "         | train numerical health: "
            "dead_mask=%.2f  z_std=%.3f  khat=[%.3f,%.3f]  "
            "batch_sep_dist=%.3f  attn_H=%.3f  nan_steps=%.0f  spike_steps=%.0f  var=%.5f",
            train_m["dead_ratio"], train_m["z_std"],
            train_m["khat_min"],   train_m["khat_max"],
            train_m["batch_sep_dist"],
            train_m.get("attn_entropy", float("nan")),
            train_m["n_nan_loss"],  train_m["n_loss_spike"],
            train_m.get("var_loss", 0.0),
        )
        # Warn if z_std is collapsing despite variance penalty.
        # Thresholds are relative to z_std_target so they scale correctly if target changes.
        _z_std_target  = float(cfg["vae"].get("z_std_target", 0.60))
        _z_std_warn    = _z_std_target * 0.25   # <25% of target -> warning
        _z_std_critical = _z_std_target * 0.083  # <8.3% of target -> critical (~=0.05 at default)
        if train_m["z_std"] < _z_std_warn:
            log.warning(
                "  [!] z_std=%.3f < %.3f (25%% of z_std_target=%.2f) -- latent variance collapsing. "
                "Raise z_var_weight (currently %.1f) or lower spectral_weight.",
                train_m["z_std"], _z_std_warn, _z_std_target,
                float(cfg["vae"].get("z_var_weight", 3.0)),
            )

        # -- per-epoch subtle failure checks -----------------------------------
        if train_m["n_nan_loss"] > 0:
            log.warning(
                "  [!!] %d NaN/Inf loss steps this epoch -- "
                "gradient explosion or corrupted batch. Reduce lr or clip harder.",
                int(train_m["n_nan_loss"]),
            )
        if train_m["n_loss_spike"] > 0:
            log.warning(
                "  [!] %d loss spike steps this epoch -- "
                "training instability. Check batch normalisation and data pipeline.",
                int(train_m["n_loss_spike"]),
            )
        if train_m["dead_ratio"] > 0.90:
            log.warning(
                "  [!] dead_mask=%.2f -- >90%% of timesteps are masked. "
                "Data pipeline may be producing near-empty profiles.",
                train_m["dead_ratio"],
            )
        if train_m["z_std"] < _z_std_critical:
            log.warning(
                "  [!!] z_std=%.4f < %.4f (critical, <8%% of z_std_target=%.2f) -- "
                "encoder producing near-constant latent. "
                "Check weight initialisation, lr, and that input is not constant.",
                train_m["z_std"], _z_std_critical, _z_std_target,
            )
        if train_m["batch_sep_dist"] < 0.1 and epoch >= 5:
            log.warning(
                "  [!] batch_sep_dist=%.3f < 0.1 after epoch %d -- "
                "clear/cloudy latents not separating even within batches. "
                "Regime sep loss may not be receiving correct mean_k labels.",
                train_m["batch_sep_dist"], epoch + 1,
            )
        if train_m["grad_norm"] > 10.0:
            log.warning(
                "  [!] gnorm=%.3f > 10 -- consider tightening grad_clip (current=%.1f).",
                train_m["grad_norm"], tc["grad_clip"],
            )
        # Stagnation: val recon not improved for >half patience despite loss still high.
        # Sub-patience threshold is half of early_stopping_patience so the warning fires
        # early enough to be actionable, and scales correctly if patience changes.
        _stagnation_patience = max(1, tc["early_stopping_patience"] // 2)
        _stagnation_recon_thr = float(cfg["vae"].get("vae_stagnation_recon_thr", 0.15))
        if epochs_no_imp >= _stagnation_patience and val_m["recon"] > _stagnation_recon_thr:
            log.warning(
                "  [!] No val improvement for %d epochs and val_recon=%.5f > %.3f (stagnation threshold) -- "
                "possible LR too low or latent_dim too small.",
                epochs_no_imp, val_m["recon"], _stagnation_recon_thr,
            )

        # Warn only when sep loss has never fired AND separation is still insufficient.
        # fired=0% when sep_dist >= margin is correct - the loss goes silent once satisfied.
        if epoch == 9 and train_m["sep_fired_pct"] == 0.0:
            sep_dist_now = train_m.get("batch_sep_dist", float("inf"))
            # P7/P11: margin is now dynamic (alpha x observed d_co); use alpha as proxy check.
            alpha = float(cfg["vae"].get("dynamic_margin_alpha", 0.70))
            if sep_dist_now < 0.5:   # sanity floor - d_co < 0.5 means latents fully collapsed
                log.warning(
                    "  [!] Epoch 10: sep loss has fired=0%% for all 10 epochs and "
                    "batch_sep_dist=%.3f is very small (alpha=%.2f). "
                    "Check cfg['vae']['regime_sep_weight'] > 0 and that latent variance "
                    "penalty is active (z_std_target=%.2f, z_var_weight=%.1f).",
                    sep_dist_now, alpha,
                    float(cfg["vae"].get("z_std_target", 0.60)),
                    float(cfg["vae"].get("z_var_weight", 3.0)),
                )

        if (epoch + 1) % diag_freq == 0:
            diag = _vae_val_diagnostics(model, val_loader, device, cfg)
            _log_vae_diagnostics(diag, val_m["recon"], cfg)
            if is_best:
                best_diag = diag

        if (epoch + 1) % grad_log_freq == 0:
            _log_per_layer_grads_from_loader(
                model, train_loader, optim, device, cfg, epoch, "AE"
            )

        # -- Adaptive controller: adjust cfg in-place based on health signals --
        # Called after the diag block so it can use the val-set latent_std when
        # available.  tc["grad_clip"] is re-read from cfg each epoch so any
        # tightening takes effect on the very next call to _run_vae_epoch.
        _ctrl_diag = diag if (epoch + 1) % diag_freq == 0 else {}
        _ctrl_actions = _vae_controller.step_vae(epoch, train_m, val_m, _ctrl_diag)

        # Apply controller side-effects that require live PyTorch objects.
        # The controller mutates cfg in-place; two rules need extra work:
        #
        # 1. nan_guard: sets tc["lr"] but the AdamW optimizer never re-reads
        #    cfg after construction.  Propagate the new LR to all param groups.
        # 2. attn_entropy: sets cfg["vae"]["encoder_dropout"] but nn.Dropout.p
        #    is fixed at construction.  Patch every Dropout module in the
        #    encoder directly so the change takes effect next forward pass.
        for _a in _ctrl_actions:
            if _a.rule == "nan_guard":
                for _pg in optim.param_groups:
                    _pg["lr"] = _a.new_val
                log.info("[CTRL] nan_guard: patched optimizer LR -> %.3e", _a.new_val)
            elif _a.rule == "attn_entropy":
                _new_p = float(cfg["vae"]["encoder_dropout"])
                for _m in model.encoder.modules():
                    if isinstance(_m, torch.nn.Dropout):
                        _m.p = _new_p
                log.info("[CTRL] attn_entropy: patched encoder Dropout.p -> %.3f", _new_p)

        # H2: Detect the dangerous sep_weight-cut + z_std-below-target interaction.
        # When the controller cuts sep_weight (correct - sep was dominating), if z_std
        # is already below z_std_target the only remaining resistance to collapse is
        # z_var_weight. If that is too low, collapse accelerates unresisted.
        # This warning fires immediately so the operator can act before collapse deepens.
        _sep_was_cut  = any(getattr(_a, "rule", "") == "sep_weight" for _a in _ctrl_actions)
        _z_std_target_h2 = float(cfg["vae"].get("z_std_target", 0.60))
        _z_std_below  = _cur_z_std < _z_std_target_h2
        if _sep_was_cut and _z_std_below:
            log.warning(
                "  [H2] DANGER: CTRL cut sep_weight this epoch (now %.3f) AND "
                "z_std=%.3f < target=%.3f. Removing sep pressure while z_std is "
                "collapsing accelerates latent collapse. "
                "ACTION: raise z_var_weight (currently %.2f) by 1.5x immediately. "
                "The controller will also attempt a boost next epoch if z_std stays low.",
                float(cfg["vae"].get("regime_sep_weight", 1.5)),
                _cur_z_std, _z_std_target_h2,
                float(cfg["vae"].get("z_var_weight", 1.0)),
            )

        # -- JSON epoch record -------------------------------------------------
        epoch_record: Dict = {
            "stage": "vae",
            "epoch": epoch + 1,
            "is_best": bool(is_best),
            "lr": float(optim.param_groups[0]["lr"]),
            # live controller parameter snapshot (allows post-hoc analysis of what fired)
            "ctrl_grad_clip":         float(tc["grad_clip"]),
            "ctrl_sep_weight":        float(cfg["vae"].get("regime_sep_weight", 2.0)),
            "ctrl_alpha":             float(cfg["vae"].get("dynamic_margin_alpha", 0.70)),
            "ctrl_z_var_weight":      float(cfg["vae"].get("z_var_weight", 3.0)),
            "ctrl_z_var_weight_upper": float(cfg["vae"].get("z_var_weight_upper", 0.0)),
            # train metrics
            "train_recon":        train_m["recon"],
            "train_spectral":     train_m["spectral"],
            "train_sep":          train_m["sep"],
            "train_var_loss":     train_m.get("var_loss", 0.0),
            "train_total":        train_m["total"],
            "train_grad_norm":    train_m["grad_norm"],
            "train_sep_fired_pct": train_m["sep_fired_pct"],
            "train_z_std":        train_m["z_std"],
            "train_batch_sep_dist": train_m["batch_sep_dist"],
            "train_attn_entropy": train_m.get("attn_entropy", float("nan")),
            "train_dead_ratio":   train_m["dead_ratio"],
            "train_khat_min":     train_m["khat_min"],
            "train_khat_max":     train_m["khat_max"],
            "train_n_nan_loss":   train_m["n_nan_loss"],
            "train_n_loss_spike": train_m["n_loss_spike"],
            # val metrics
            "val_recon":          val_m["recon"],
            "val_spectral":       val_m["spectral"],
            "val_total":          val_m["total"],
            "val_z_std":          val_m["z_std"],
            "val_batch_sep_dist": val_m["batch_sep_dist"],
            "param_norm":         float(param_norm),
            "epochs_no_imp":      epochs_no_imp,
        }
        # Attach diag block if it was computed this epoch
        if (epoch + 1) % diag_freq == 0:
            _n_regimes    = int(cfg["diffusion"].get("n_regimes", 4))
            _regime_names = cfg["data"]["regime"].get(
                "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
            )[:_n_regimes]
            _diag_dynamic: Dict[str, float] = {
                "diag_ramp_mae":           diag.get("ramp_mae",       float("nan")),
                "diag_latent_mean":        diag.get("latent_mean",    float("nan")),
                "diag_latent_std":         diag.get("latent_std",     float("nan")),
                "diag_latent_n_dead_dims": diag.get("latent_n_dead_dims", float("nan")),
                "diag_latent_n_hot_dims":  diag.get("latent_n_hot_dims",  float("nan")),
                "diag_decoder_clamp_rate": diag.get("decoder_clamp_rate", float("nan")),
                "diag_latent_sep_dist":    diag.get("latent_sep_dist",    float("nan")),
            }
            for _nm in _regime_names:
                _diag_dynamic[f"diag_recon_{_nm}"]    = diag.get(f"recon_{_nm}",    float("nan"))
                _diag_dynamic[f"diag_ramp_mae_{_nm}"] = diag.get(f"ramp_mae_{_nm}", float("nan"))
                _diag_dynamic[f"diag_n_{_nm}"]        = diag.get(f"latent_n_{_nm}", float("nan"))
            # pairwise sep distances
            for _i in range(len(_regime_names)):
                for _j in range(_i + 1, len(_regime_names)):
                    _key = f"latent_sep_{_regime_names[_i]}_{_regime_names[_j]}"
                    _diag_dynamic[f"diag_sep_{_regime_names[_i]}_{_regime_names[_j]}"] = diag.get(_key, float("nan"))
            epoch_record.update(_diag_dynamic)
        epoch_record.update(ctrl_actions_to_record(_ctrl_actions))
        _append_json_log(vae_jsonlog, epoch_record)

        if epochs_no_imp >= patience:
            log.info("Early stopping at epoch %d (patience=%d).", epoch + 1, patience)
            break

    load_checkpoint(best_path, model)
    if not best_diag:
        best_diag = _vae_val_diagnostics(model, val_loader, device, cfg)
    _vae_action_guide(best_diag, best_val, cfg)
    log.info("VAE training complete. Best val_recon: %.5f", best_val)
    return model


# =============================================================================
# Latent cache + tau-transform fitting
# =============================================================================

@torch.no_grad()
def build_latent_cache(
    vae: SolarVAE,
    loader: DataLoader,
    device: torch.device,
    cache_path: str | Path,
    cfg: Optional[Dict] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, Optional[List[str]]]:
    #Encode all training days to z.
    # 
    # Returns (latents, regime_labels, pass0_n, pass0_dates).
    # 
    # latents       : (N_total, d_z) float32 - all aug passes concatenated.
    # regime_labels : (pass0_n,) int8 - GMM labels from pass 0 only (deterministic).
    #                 None if batch["regime"] is absent.
    # pass0_n       : number of latents from the deterministic eval-mode pass (pass 0).
    #                 Used by fit_latent_tau_transform to restrict centroid calibration
    #                 to pass-0 only, avoiding variance inflation from stochastic passes.
    # pass0_dates   : list[str] of length pass0_n - the date string for each pass-0
    #                 latent, collected in DataLoader iteration order.  Used by
    #                 relabel_latents_kmeans to build a date-keyed label map so that
    #                 dataset.py can look up labels by date rather than by position
    #                 (position-based lookup was broken because the loader is shuffled
    #                 with replacement, so its iteration order differs from any sorted
    #                 profile list).
    #
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    n_passes = 1
    if cfg is not None:
        n_passes = int(cfg["training"]["diffusion"]["latent_cache_aug_passes"])

    vae.eval()   # pass 0 (first pass): eval mode -> deterministic, no dropout
    all_z       = []
    all_regimes = []
    all_dates: List[str] = []   # pass-0 date strings, one per (batch, window) slot
    pass0_n = 0  # number of latents from deterministic pass 0 (for calibration)
    for pass_idx in range(n_passes):
        # Bug fix: previously vae.eval() was set once before the loop, which
        # disabled encoder_dropout for all passes. With dropout off, every pass
        # produces identical latents - the aug passes added data volume but zero
        # stochastic diversity. The tau-transform was then fitted on a redundant
        # dataset that underestimated true latent variance, miscalibrating tau-space
        # scale for all of Stage 2.
        # Fix: pass 0 uses eval() for a clean deterministic baseline; passes 1+
        # use train() so encoder_dropout is active, each pass produces genuinely
        # different latents for the same day. torch.no_grad() is kept throughout
        # so memory and speed are unaffected.
        if pass_idx == 0:
            vae.eval()
        else:
            vae.train()   # activates encoder Dropout for stochastic aug passes
        pass_z = []
        for batch in loader:
            B, W, T_max = batch["k_star"].shape
            _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7
            k_star = batch["k_star"].to(device).reshape(B * W, T_max)
            phys   = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
            mask   = batch["valid_mask"].to(device).reshape(B * W, T_max)
            obs    = _get_obs_phys(batch, device, B, W, T_max)
            z, _ = vae.encode(k_star, phys, mask, obs)
            pass_z.append(z.cpu().numpy())
            if pass_idx == 0:
                if "regime" in batch:
                    all_regimes.append(batch["regime"].numpy().reshape(B * W))
                # Collect dates in the same flattened order as the latents.
                # batch["date"] is list[list[str]] of shape (B, W).
                if "date" in batch:
                    for b in range(B):
                        for w in range(W):
                            all_dates.append(str(batch["date"][b][w]))
        all_z.extend(pass_z)
        if pass_idx == 0:
            # pass0_n must count UNIQUE window-slots, not total sampled slots.
            # WeightedRandomSampler samples with replacement, so the same day
            # can appear multiple times in pass_z - inflating density estimates
            # near the mean and miscalibrating percentile bounds in
            # fit_latent_tau_transform.
            # We reconstruct a deduplicated index by keeping only the FIRST
            # occurrence of each date (in iteration order) so latents[:pass0_n]
            # contains each unique day exactly once.
            if all_dates:
                _seen: set = set()
                _unique_idx: List[int] = []
                for _i, _d in enumerate(all_dates):
                    if _d not in _seen:
                        _seen.add(_d)
                        _unique_idx.append(_i)
                pass0_n = len(_unique_idx)
                # Reorder the first pass0_n rows of all_z to be the unique ones.
                # all_z currently holds pass_z arrays; concatenate pass_z and
                # replace in-place with the deduplicated subset.
                _pass0_concat = np.concatenate(pass_z, axis=0)
                all_z[-len(pass_z):] = []          # remove the originals
                all_z.append(_pass0_concat[_unique_idx])   # add deduped
                all_dates = [all_dates[i] for i in _unique_idx]
                # Regimes were appended per-slot inside the batch loop; realign.
                if all_regimes:
                    _reg_concat = np.concatenate(all_regimes)
                    all_regimes = [_reg_concat[_unique_idx]]
            else:
                pass0_n = sum(a.shape[0] for a in pass_z)
        log.info("Latent cache pass %d/%d complete", pass_idx + 1, n_passes)

    vae.eval()   # restore eval mode - leave model in clean state for caller
    latents = np.concatenate(all_z, axis=0)
    np.save(cache_path, latents)
    log.info(
        "Latent cache saved -> %s  shape=%s  (n_aug_passes=%d)",
        cache_path, latents.shape, n_passes,
    )
    log.info(
        "Latent stats | mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        latents.mean(), latents.std(), latents.min(), latents.max(),
    )
    # Regimes and dates collected only from pass 0 (deterministic).
    regimes = np.concatenate(all_regimes).astype(np.int8) if all_regimes else None
    pass0_dates: Optional[List[str]] = all_dates if all_dates else None
    return latents, regimes, pass0_n, pass0_dates


def relabel_latents_kmeans(
    latents: np.ndarray,
    gmm_labels: np.ndarray,
    cfg: Dict,
    pass0_dates: Optional[List[str]] = None,
) -> np.ndarray:
    #Re-cluster pass-0 latents with k-means, then map clusters to regime labels.
    # 
    # After Stage 1 the encoder has learned its own geometry that may differ from
    # what the frozen GMM saw in K*/variability feature space.  Using GMM labels
    # directly for tau-transform calibration means centroids are fitted in a space
    # that the denoiser organised differently.  This creates the misalignment that
    # causes the wrong transition matrix and wrong regime frequency at generation time.
    # 
    # Steps:
    #   1. Run k-means on pass-0 latents (same n_components as GMM).
    #   2. Map each k-means cluster -> GMM regime label by majority vote, preserving
    #      semantic meaning (clear=0, mixed=1/2, overcast=3).
    #   3. Return new per-sample labels aligned with the latent geometry.
    #   4. Save to cache/latent_relabels.npy (positional, for fit_latent_tau_transform)
    #      and cache/latent_relabels_map.json ({date: label}), which is what
    #      dataset.py uses to apply relabels without any positional alignment issue.
    # 
    # Parameters
    # ----------
    # latents     : (N, d_z) float32 - pass-0 latents only (not aug passes).
    # gmm_labels  : (N,) int8 - original GMM labels aligned with latents.
    # cfg         : full config dict.
    # pass0_dates : list[str] of length N - date string for each latent, collected
    #               by build_latent_cache in DataLoader iteration order.  When
    #               provided, a date-keyed JSON map is saved alongside the .npy so
    #               dataset.py can look up the correct label for any day without
    #               depending on positional alignment between the cache and the
    #               sorted profile list.
    # 
    # Returns
    # -------
    # new_labels : (N,) int8 relabelled regime IDs aligned with latent geometry.
    #              Falls back to gmm_labels unchanged if any regime goes missing.
    #
    from sklearn.cluster import KMeans

    n_comp       = int(cfg["data"]["regime"].get("n_components", 4))
    random_state = int(cfg["data"]["regime"].get("kmeans_random_state", 42))
    n_init_km    = int(cfg["data"]["regime"].get("kmeans_n_init", 10))

    log.info(
        "relabel_latents_kmeans: fitting k-means (k=%d, n_init=%d) on %d pass-0 latents ...",
        n_comp, n_init_km, len(latents),
    )

    km = KMeans(
        n_clusters=n_comp, n_init=n_init_km,
        random_state=random_state, max_iter=500,
    )
    km.fit(latents)
    cluster_ids = km.labels_   # (N,) int - 0 ... n_comp-1

    # Map each cluster -> regime label by majority vote from gmm_labels.
    cluster_to_regime = np.empty(n_comp, dtype=np.int64)
    for ci in range(n_comp):
        mask = (cluster_ids == ci)
        if mask.sum() == 0:
            cluster_to_regime[ci] = ci % n_comp
            log.warning("relabel_latents_kmeans: cluster %d is empty - assigning regime %d.", ci, ci % n_comp)
            continue
        votes  = gmm_labels[mask].astype(np.int64)
        counts = np.bincount(votes, minlength=n_comp)
        cluster_to_regime[ci] = int(counts.argmax())

    # Guard: every regime must be assigned exactly once.
    # If two clusters vote for the same regime, the latent space is not fully
    # separated - fall back to GMM labels so downstream is not broken.
    assigned = set(cluster_to_regime.tolist())
    missing  = set(range(n_comp)) - assigned
    if missing:
        log.warning(
            "relabel_latents_kmeans: regimes %s not assigned (two clusters voted for the "
            "same regime).  Latent space may not be fully separated yet - "
            "falling back to GMM labels.  Consider higher regime_sep_weight or "
            "longer Stage 1 training.",
            sorted(missing),
        )
        return gmm_labels.copy()

    # Check uniqueness - each regime must appear exactly once in the mapping.
    if len(assigned) != n_comp:
        log.warning(
            "relabel_latents_kmeans: mapping is not injective (%d unique assignments for "
            "%d regimes) - falling back to GMM labels.",
            len(assigned), n_comp,
        )
        return gmm_labels.copy()

    new_labels = cluster_to_regime[cluster_ids].astype(np.int8)   # (N,)

    # Quality metric: agreement with GMM labels.
    agreement = float((new_labels == gmm_labels).mean())
    log.info(
        "relabel_latents_kmeans complete | cluster->regime: %s | "
        "agreement with GMM: %.1f%%",
        {ci: int(cluster_to_regime[ci]) for ci in range(n_comp)},
        100.0 * agreement,
    )
    if agreement < 0.50:
        log.warning(
            "  [!!] k-means relabelling agreement=%.1f%% < 50%%.  "
            "The latent geometry differs substantially from GMM feature space.  "
            "Regime conditioning in Stage 2 will use latent-aligned labels - "
            "verify the generated transition matrix after training.",
            100.0 * agreement,
        )

    # -- Persist relabels ------------------------------------------------------
    relabel_path = Path(cfg.get("paths", {}).get("latent_relabels", "cache/latent_relabels.npy"))
    relabel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(relabel_path, new_labels)
    log.info("Latent relabels saved -> %s  shape=%s", relabel_path, new_labels.shape)

    # Save a date-keyed JSON map alongside the positional .npy.
    # This is what dataset.py loads: it looks up label by date string, avoiding
    # any positional alignment issue between the shuffled DataLoader iteration
    # order and the sorted profile list in build_datasets().
    if pass0_dates is not None and len(pass0_dates) == len(new_labels):
        import json as _json
        # When the same date appears multiple times (overlapping windows with
        # WeightedRandomSampler replacement), take the majority vote across all
        # occurrences so the final label is the most representative one.
        from collections import Counter as _Counter
        date_vote: Dict[str, list] = {}
        for d, lbl in zip(pass0_dates, new_labels.tolist()):
            date_vote.setdefault(d, []).append(int(lbl))
        date_map: Dict[str, int] = {
            d: int(_Counter(votes).most_common(1)[0][0])
            for d, votes in date_vote.items()
        }
        map_path = relabel_path.with_suffix(".json").with_name(
            relabel_path.stem + "_map.json"
        )
        map_path.write_text(_json.dumps(date_map, separators=(",", ":")))
        log.info(
            "Latent relabels date-map saved -> %s  (%d unique dates)",
            map_path, len(date_map),
        )
        # Note 5 FIX: use direct assignment instead of setdefault.
        # setdefault("paths", {}) creates a throwaway dict when "paths" is
        # absent, making the assignment invisible to callers. Direct assignment
        # always updates the correct dict. cfg["paths"] always exists when
        # invoked from main.py, but direct assignment is unconditionally safe.
        cfg["paths"]["latent_relabels_map"] = str(map_path)
    else:
        if pass0_dates is None:
            log.warning(
                "relabel_latents_kmeans: pass0_dates not provided - "
                "date-keyed map not saved. dataset.py will fall back to "
                "positional .npy alignment, which may be incorrect when the "
                "DataLoader uses a shuffled sampler. Pass pass0_dates from "
                "build_latent_cache to enable correct relabelling."
            )
        else:
            log.warning(
                "relabel_latents_kmeans: pass0_dates length (%d) != new_labels "
                "length (%d) - date-keyed map not saved.",
                len(pass0_dates), len(new_labels),
            )

    return new_labels


def _gmm_fallback_centroids(
    cfg: Dict,
    transform: "LatentTauTransform",
    n_regimes: int,
) -> list:
    #Compute class_tau_centroids from GMM K* means when latent labels are unavailable.
    # 
    # Projects each GMM component's mean K* through Beer-Lambert:
    #     tau = -log(mean_K* / k_max)
    # then maps them to regime classes via component_to_regime.  This is
    # geometrically consistent with LatentTauTransform (which applies the same
    # log map after per-dim normalisation) and always produces valid, monotone
    # centroids so the regime feedback loop is never silently disabled.
    # 
    # Called as a fallback when regime_labels is None or sample counts are
    # insufficient.  Empirical centroids from real latents (the normal path) are
    # always preferred when available.
    #
    import json as _json
    k_max           = float(cfg["physics"]["k_max"])
    z_norm_clamp_lo = float(getattr(transform, "z_norm_clamp_lo", 0.01))
    gmm_path        = cfg["paths"].get("regime_gmm", None)

    if gmm_path is None or not Path(gmm_path).exists():
        # No GMM available - return evenly-spaced centroids between 0 and tau_max
        tau_max = -np.log(z_norm_clamp_lo)
        centroids = [round(tau_max * i / max(n_regimes - 1, 1), 4) for i in range(n_regimes)]
        log.warning(
            "  _gmm_fallback_centroids: GMM not found at '%s'. "
            "Using evenly-spaced fallback centroids: %s", gmm_path, centroids,
        )
        return centroids

    with open(gmm_path) as _f:
        gmm = _json.load(_f)

    component_to_regime = gmm["component_to_regime"]
    means_kstar         = gmm["means"]   # component means in K* feature space

    centroids = [None] * n_regimes
    for comp_idx, regime_class in enumerate(component_to_regime):
        if regime_class >= n_regimes:
            continue
        mean_k   = float(means_kstar[comp_idx][0])
        fraction = min(max(mean_k / k_max, z_norm_clamp_lo), 1.0)
        centroids[regime_class] = round(-np.log(fraction), 4)

    # Fill any gaps (regime class not covered by any component)
    tau_max = -np.log(z_norm_clamp_lo)
    for i in range(n_regimes):
        if centroids[i] is None:
            centroids[i] = round(tau_max * i / max(n_regimes - 1, 1), 4)
            log.warning(
                "  _gmm_fallback_centroids: no GMM component maps to regime %d - "
                "using linear interpolation: tau=%.4f", i, centroids[i],
            )

    log.info(
        "  _gmm_fallback_centroids: GMM-derived centroids (regimes 0..%d): %s",
        n_regimes - 1, centroids,
    )
    return centroids


def fit_latent_tau_transform(
    latents: np.ndarray,
    cfg: Dict,
    regime_labels: Optional[np.ndarray] = None,
    pass0_n: Optional[int] = None,
) -> LatentTauTransform:
    #Fit per-dimension tau transform on training latents with polarity correction.
    # 
    # regime_labels : (N,) int8 from build_latent_cache (0=clear, 1=mixed, 2=overcast).
    # When provided, dims where mean_z[clear] < mean_z[cloudy] are reflected so
    # that high-z = clear universally before the tau map is applied.
    # Without this the forward diffusion noises toward the wrong regime prior on
    # the inverted dims, and the denoiser learns the wrong sign for those directions.
    # 
    # pass0_n : number of latents from the deterministic pass 0.  When given, the
    # tau-space separation calibration check uses only the pass-0 slice to avoid
    # inflated variance from stochastic aug passes biasing the centroid estimate.
    # 
    # After fitting, logs the tau-space centroid separation so you can verify it
    # exceeds the Stage-2 tau_separation threshold (0.20).  The z-space sep margin
    # (regime_sep_margin in vae config) and the tau-space threshold are on different
    # scales - this check bridges the gap so a passing z-space sep loss still
    # guarantees a passing tau_separation at Stage 2 startup.
    #
    vc        = cfg["vae"]
    k_max     = cfg["physics"]["k_max"]
    p_lo      = float(vc["latent_tau_percentile_lo"])
    p_hi      = float(vc["latent_tau_percentile_hi"])
    std_m     = float(vc["latent_tau_std_margin"])
    transform = LatentTauTransform(
        k_max=k_max,
        z_norm_clamp_lo=float(vc.get("z_norm_clamp_lo", 0.01)),
    )


    # Use only pass-0 (deterministic, eval-mode) latents for fitting bounds.
    # Aug passes (passes 1+) are stochastic (encoder dropout active) and inflate
    # per-dim variance, which makes z_min/z_max too wide.  A too-wide transform
    # compresses tau-space separation and miscalibrates the diffusion prior.
    # regime_labels already align with pass-0 only (collected in build_latent_cache).
    if pass0_n is not None and pass0_n > 0 and pass0_n <= len(latents):
        fit_latents       = latents[:pass0_n]
        fit_regime_labels = regime_labels  # already pass-0 only
        log.info(
            "fit_latent_tau_transform: fitting on pass-0 latents only (%d / %d total)",
            pass0_n, len(latents),
        )
    else:
        fit_latents       = latents
        fit_regime_labels = regime_labels

    transform.fit(
        fit_latents,
        regime_labels=fit_regime_labels,
        percentile_lo=p_lo,
        percentile_hi=p_hi,
        std_margin_factor=std_m,
        polarity_min_samples=int(cfg["vae"].get("tau_cal_min_samples", 10)),
    )
    transform.save(cfg["paths"]["latent_tau_stats"])

    # -- Calibration check: z-space sep margin vs tau-space sep threshold ------
    # regime_separation_loss pushes centroid distance in z-space to >= margin.
    # But Stage 2 checks tau_separation >= 0.20 in tau-space.  The tau transform
    # is a nonlinear map, so the two thresholds are not directly comparable.
    # We compute the actual tau-space separation here and warn if it's below 0.20
    # so the user knows whether to raise regime_sep_margin before Stage 2.
    import torch as _torch
    import json as _json

    n_regimes      = int(cfg["diffusion"].get("n_regimes", 4))
    tau_stats_path = cfg["paths"]["latent_tau_stats"]
    _tau_cal_min   = int(cfg["vae"].get("tau_cal_min_samples", 10))

    _expected_len = pass0_n if (pass0_n is not None and pass0_n > 0) else len(latents)
    if regime_labels is not None and len(regime_labels) == _expected_len:
        # Use only pass-0 (deterministic) latents.
        # Aug passes inflate variance via stochastic dropout; regime_labels are
        # from pass 0 only so they align with latents[:pass0_n].
        n_cal      = pass0_n if (pass0_n is not None and pass0_n <= len(latents)) else len(latents)
        cal_z      = latents[:n_cal]
        cal_labels = regime_labels  # already pass-0 only

        is_clear  = (cal_labels == 0)
        is_cloudy = (cal_labels >= 1)

        if is_clear.sum() >= _tau_cal_min and is_cloudy.sum() >= _tau_cal_min:
            z_t     = _torch.from_numpy(cal_z.astype(np.float32))
            tau_all = transform.to_tau_space(z_t).numpy()
            tau_sep = float(tau_all[is_cloudy].mean(axis=0).mean()
                            - tau_all[is_clear].mean(axis=0).mean())
            tau_threshold = float(cfg["diagnostics"]["tau_separation_min_to_proceed"])
            if tau_sep < tau_threshold:
                log.warning(
                    "  [!!] tau_separation=%.4f < %.2f after fitting tau transform. "
                    "Raise cfg['vae']['dynamic_margin_alpha'] or regime_sep_weight "
                    "and retrain Stage 1 before proceeding to Stage 2.",
                    tau_sep, tau_threshold,
                )
            else:
                log.info(
                    "  tau_separation=%.4f >= %.2f - tau-space separation healthy.",
                    tau_sep, tau_threshold,
                )

            # -- Compute per-class tau centroids from real encoded latents ----------
            # generate.py uses these to assign regime labels by nearest centroid,
            # eliminating any need for hardcoded K* thresholds at inference.
            class_tau_centroids = []
            for cls_idx in range(n_regimes):
                mask = (cal_labels == cls_idx)
                if mask.sum() >= 2:
                    centroid = float(tau_all[mask].mean(axis=0).mean())
                else:
                    # Interpolate missing class centroid from global mean
                    centroid = float(tau_all.mean(axis=0).mean())
                    log.warning(
                        "  fit_latent_tau_transform: class %d has only %d samples "
                        "- centroid interpolated from global mean.", cls_idx, int(mask.sum()),
                    )
                class_tau_centroids.append(centroid)

        else:
            # Not enough labelled samples per class - fall back to GMM-based centroids
            # so class_tau_centroids is always written and the regime feedback loop
            # is never silently disabled at generation time.
            log.warning(
                "  fit_latent_tau_transform: insufficient samples for empirical centroids "
                "(clear=%d, cloudy=%d, min=%d). "
                "Falling back to GMM-based tau centroids.",
                int(is_clear.sum()), int(is_cloudy.sum()), _tau_cal_min,
            )
            class_tau_centroids = _gmm_fallback_centroids(cfg, transform, n_regimes)

    else:
        # regime_labels not provided (or length mismatch) - fall back to GMM-based
        # centroids so class_tau_centroids is ALWAYS written and the regime feedback
        # loop is never silently disabled at generation time.
        log.warning(
            "  fit_latent_tau_transform: regime_labels not provided (or length mismatch). "
            "Falling back to GMM-based tau centroids for class_tau_centroids. "
            "Pass regime_labels from build_latent_cache for empirical centroids.",
        )
        class_tau_centroids = _gmm_fallback_centroids(cfg, transform, n_regimes)

    # -- Validate monotone ordering --------------------------------------------
    # clear=0 should have smallest tau, overcast=n-1 largest tau.
    for _ci in range(len(class_tau_centroids) - 1):
        if class_tau_centroids[_ci] >= class_tau_centroids[_ci + 1]:
            log.warning(
                "  [!] class_tau_centroids are NOT monotonically ascending "
                "in label-index order: %s. "
                "Regime label %d (tau=%.4f) >= label %d (tau=%.4f). "
                "The nearest-centroid feedback loop in generate.py will "
                "still work correctly, but verify that k-means relabelling "
                "preserved the clear=low-tau / overcast=high-tau ordering.",
                [round(v, 4) for v in class_tau_centroids],
                _ci, class_tau_centroids[_ci],
                _ci + 1, class_tau_centroids[_ci + 1],
            )
            break

    # -- Write centroids into latent_tau_stats.json ----------------------------
    with open(tau_stats_path) as _f:
        blob = _json.load(_f)
    blob["class_tau_centroids"] = class_tau_centroids
    # Global training-set regime frequencies - used by generate.py to seed
    # the initial noise distribution.  Stored alongside class_tau_centroids
    # so generation works without access to the original training dataset.
    if regime_labels is not None and len(regime_labels) > 0:
        _counts = np.bincount(
            regime_labels.astype(int), minlength=n_regimes
        ).astype(np.float32)
        blob["global_regime_freqs"] = (_counts / _counts.sum()).tolist()
        log.info(
            "  Global regime frequencies written: %s",
            [round(v, 4) for v in blob["global_regime_freqs"]],
        )
    with open(tau_stats_path, "w") as _f:
        _json.dump(blob, _f, indent=2)
    log.info(
        "  Per-class tau centroids written to %s: %s",
        tau_stats_path, [round(v, 4) for v in class_tau_centroids],
    )

    return transform


# =============================================================================
# Stage 2 - diffusion epoch runner
# =============================================================================

def _run_diff_epoch(
    model: SolarDenoiser,
    vae: SolarVAE,
    loader: DataLoader,
    optim,
    schedule: TauNoiseSchedule,
    latent_transform: LatentTauTransform,
    device: torch.device,
    cfg: Dict,
    grad_clip: float,
    log_every: int,
    train: bool,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    #Run one diffusion epoch.
    # 
    # Returns loss / loss_clear / loss_cloudy / snr_mean / snr_std / grad_norm.
    #
    dc         = cfg["diffusion"]
    cfg_p      = dc["cfg_dropout_prob"]
    pred_type  = dc["prediction_type"]
    min_snr_g  = dc["min_snr_gamma"]
    # Per-class loss weights indexed by GMM regime label [clear, mx-clear, mx-oc, overcast].
    # These are independent of the WeightedRandomSampler; keeping them close to [1,1.5,2,2.5]
    # avoids double-compensation while still providing gradient emphasis for rarer regimes.
    regime_loss_weights_diff = dc["regime_loss_weights_diff"]
    per_day_k  = bool(dc.get("per_day_k_sampling", True))
    inp_prob   = float(dc.get("inpainting_prob", 0.0))
    n_ctx_tr   = int(dc["n_ctx_train"])
    _tau_overflow_mean_thr = float(cfg["diagnostics"].get("tau_overflow_mean_thr", 3.0))
    _tau_overflow_std_thr  = float(cfg["diagnostics"].get("tau_overflow_std_thr",  4.0))

    total_loss     = 0.0
    total_clear    = 0.0; n_clear  = 0
    total_cloudy   = 0.0; n_cloudy = 0
    total_gnorm    = 0.0
    total_inp_loss = 0.0; n_inp = 0
    total_cloudy_days = 0; total_days = 0   # track observed regime fraction
    # Replace all_snr list with running Welford accumulators to avoid
    # unbounded CPU memory growth (list grew by B*W floats every batch).
    _snr_n    = 0
    _snr_mean = 0.0
    _snr_M2   = 0.0   # for Welford online variance
    n_batches = 0
    # -- diffusion-specific health stats --------------------------------------
    total_tau_mean  = 0.0
    total_tau_std   = 0.0
    n_tau_overflow  = 0
    total_l_temporal = 0.0
    total_l_variance = 0.0

    # -- Auxiliary loss weights (ramp coherence + variance preservation) -------
    # These were defined in config and implemented in diffusion_loss() but were
    # never read here or passed to the loss call - so they had zero effect on
    # training despite temporal_weight=0.10 and variance_weight=0.10 in config.
    # Fix: read them here and pass them through. tau0_hat is recovered from
    # v-prediction inside the loop using the standard inverse formula.
    _temporal_w  = float(dc.get("temporal_weight",  0.0))
    _variance_w  = float(dc.get("variance_weight",  0.0))
    _ramp_snr_fl = float(dc.get("ramp_snr_floor",   1.0))
    _use_aux     = train and (_temporal_w > 0.0 or _variance_w > 0.0)

    for batch in loader:
        B, W, T_max = batch["k_star"].shape
        _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7

        with torch.no_grad():
            k_star  = batch["k_star"].to(device).reshape(B * W, T_max)
            phys_2d = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
            mask_2d = batch["valid_mask"].to(device).reshape(B * W, T_max)
            obs_2d  = _get_obs_phys(batch, device, B, W, T_max)
            z, _ = vae.encode(k_star, phys_2d, mask_2d, obs_2d)
            z     = z.reshape(B, W, -1)
            tau0 = latent_transform.to_tau_space(z)

        # -- Per-day or per-window k sampling --------------------------------
        if per_day_k:
            # (B, W): each day gets an independent noise level.
            # Richer gradient signal - days at different window positions
            # see different SNR levels in a single step.
            k_step = torch.randint(0, dc["num_steps"], (B, W), device=device)
        else:
            k_step = torch.randint(0, dc["num_steps"], (B,), device=device)

        noise        = torch.randn_like(tau0)
        tau_k, noise = schedule.q_sample(tau0, k_step, noise)

        # -- tau health ------------------------------------------------------
        with torch.no_grad():
            tm = tau0.mean().item(); ts = tau0.std().item()
            total_tau_mean += tm; total_tau_std += ts
            if abs(tm) > _tau_overflow_mean_thr or ts > _tau_overflow_std_thr:
                n_tau_overflow += 1
                if n_tau_overflow == 1:
                    log.warning(
                        "  [!] tau0 out-of-range at step=%d: mean=%.3f std=%.3f -- "
                        "LatentTauTransform may not be fitted to current AE. "
                        "Rebuild latent cache and re-fit.",
                        n_batches, tm, ts,
                    )

        day_feat   = batch["day_features"].to(device)
        location   = batch["location"].to(device)
        intra_phys = batch["intraday_phys"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        drop_mask = (torch.rand(B, device=device) < cfg_p) if train else \
                    torch.zeros(B, device=device, dtype=torch.bool)

        # -- Inpainting conditioning -------------------------------------------
        # With probability inp_prob, treat this window as an inpainting task:
        # pin the first n_ctx_tr days at clean tau0 (context) and train the
        # model to denoise only the remaining W - n_ctx_tr target days.
        #
        # IMPORTANT - day_mask:
        # Context days are pinned to tau0 so their tau_k is NOT noised at k_step.
        # The v-target get_v_target(tau0, noise, k_step) is only meaningful when
        # tau_k = sqrt(ab_k)*tau0 + sqrt(1-ab_k)*noise - which is false for pinned
        # days.  Training on a wrong target for context positions teaches the model
        # to undo the pinning at inference, directly fighting the RePaint conditioning.
        # Fix: pass day_mask=(target days only) to diffusion_loss so context positions
        # contribute zero gradient.  Same mask applied to clear/cloudy ratio diagnostic.
        is_inpainting = train and inp_prob > 0 and (torch.rand(1).item() < inp_prob)
        day_mask = None   # None = all W days active (normal unconditional step)
        if is_inpainting:
            # Pin context days at clean tau0 - matches inference RePaint conditioning.
            tau_k[:, :n_ctx_tr, :] = tau0[:, :n_ctx_tr, :]
            # Mask: only the W - n_ctx_tr target days contribute to the loss.
            day_mask = torch.zeros(B, W, device=device, dtype=torch.bool)
            day_mask[:, n_ctx_tr:] = True   # target days only

        # P9: regime conditioning for denoiser
        regime_ids = batch["regime"].to(device)   # (B, W) int64

        v_pred = model(tau_k, k_step, day_feat, location,
                       intra_phys, valid_mask, drop_mask=drop_mask,
                       regime_ids=regime_ids)

        if pred_type == "v":
            target = schedule.get_v_target(tau0, noise, k_step)
        elif pred_type == "eps":
            target = noise
        elif pred_type == "x0":
            target = tau0
        else:
            raise ValueError(f"Unknown prediction_type '{pred_type}'.")

        # -- SNR weighting ----------------------------------------------------
        if per_day_k:
            # (B, W) per-day SNR
            ab_k  = schedule.alpha_bars[k_step.reshape(-1).long()].reshape(B, W)
            snr_k = ab_k / (1.0 - ab_k + 1e-8)
            # Welford online update - avoids storing the full SNR list
            for _s in snr_k.reshape(-1).detach().cpu().tolist():
                _snr_n    += 1
                _delta     = _s - _snr_mean
                _snr_mean += _delta / _snr_n
                _snr_M2   += _delta * (_s - _snr_mean)
        else:
            ab_k  = schedule.alpha_bars[k_step.long()]
            snr_k = ab_k / (1.0 - ab_k + 1e-8)
            for _s in snr_k.detach().cpu().tolist():
                _snr_n    += 1
                _delta     = _s - _snr_mean
                _snr_mean += _delta / _snr_n
                _snr_M2   += _delta * (_s - _snr_mean)

        # Per-day loss weights from the 4-class regime_loss_weights_diff list.
        with torch.no_grad():
            regime_ids_d = batch["regime"].to(device)   # (B, W) int64
            is_cloudy_day = (regime_ids_d != 0).float()  # for cloudy-fraction logging
            # H: regime_loss_weights_diff only apply during TRAINING.
            # Validation must use uniform weights (all ones) so that val_loss
            # is a weight-independent metric suitable for early stopping.
            # Evidence from logs: val_loss jumped 15% between runs purely because
            # regime weights changed - misleading the stopping criterion.
            if train:
                per_day_w = torch.ones(B, W, device=device)
                for cls_idx, w in enumerate(regime_loss_weights_diff):
                    per_day_w[regime_ids_d == cls_idx] = float(w)
            else:
                # Uniform weights for validation - always faithful to true loss
                per_day_w = torch.ones(B, W, device=device)
            if is_inpainting:
                per_day_w[:, :n_ctx_tr] = 0.0
            # Keep per_day_w as (B, W) - diffusion_loss expects day_weights=(B, W)
            # DO NOT reduce to (B,) here; the loss function does its own reduction.

        # -- tau0 recovery for aux losses (ramp + variance) ---------------------
        # Recover the denoiser's clean-tau estimate from v-prediction:
        #   tau0 = sqrtabar_k * tau_k - sqrt(1-abar_k) * v
        # This is the same formula used at inference (p_sample / p_sample_ddim).
        # Only computed when at least one aux loss weight is non-zero AND we are
        # in training mode (val pass uses weight=0 so aux is always skipped).
        tau0_hat_for_aux = None
        if _use_aux and pred_type == "v":
            # Note 1 FIX: compute tau0_hat_for_aux WITHOUT no_grad() and WITHOUT
            # .detach() so that backprop flows through v_pred into the denoiser.
            # The previous no_grad() block silently killed all gradients from
            # l_temporal and l_variance - those losses appeared non-zero in logs
            # but produced zero gradient signal for the denoiser weights.
            # The schedule lookups (sqrt_alpha_bars, sqrt_one_minus) are
            # constant tensors - they do not need gradients themselves, but they
            # must not cut the path through v_pred.
            if per_day_k:
                sab = schedule.sqrt_alpha_bars[k_step.reshape(-1).long()].reshape(B, W, 1)
                som = schedule.sqrt_one_minus[k_step.reshape(-1).long()].reshape(B, W, 1)
            else:
                sab = schedule.sqrt_alpha_bars[k_step.long()].view(-1, 1, 1)
                som = schedule.sqrt_one_minus[k_step.long()].view(-1, 1, 1)
            tau0_hat_for_aux = sab * tau_k - som * v_pred   # gradient flows through v_pred

        loss, l_temp, l_var = diffusion_loss(
            v_pred, target, snr=snr_k,
            min_snr_gamma=min_snr_g, day_weights=per_day_w,
            day_mask=day_mask,
            tau0_hat=tau0_hat_for_aux,
            tau0_true=tau0 if _use_aux else None,
            temporal_weight=_temporal_w if _use_aux else 0.0,
            variance_weight=_variance_w if _use_aux else 0.0,
            ramp_snr_floor=_ramp_snr_fl,
            snr_for_aux=snr_k,
        )

        if is_inpainting:
            total_inp_loss += loss.item()
            n_inp          += 1

        with torch.no_grad():
            # Per-day MSE attributed to clear/cloudy for the ratio diagnostic.
            # Vectorised - no Python loop over BxW (was O(B*W) Python overhead per batch).
            raw_per_day = (v_pred - target).pow(2).mean(dim=2)   # (B, W)
            is_clear_day  = (regime_ids_d == 0)                   # (B, W) bool
            is_cloudy_day = ~is_clear_day                         # (B, W) bool
            if is_inpainting:
                ctx_mask = torch.zeros(B, W, dtype=torch.bool, device=device)
                ctx_mask[:, :n_ctx_tr] = True
                is_clear_day  = is_clear_day  & ~ctx_mask
                is_cloudy_day = is_cloudy_day & ~ctx_mask
            total_clear  += raw_per_day[is_clear_day].sum().item()
            n_clear      += int(is_clear_day.sum().item())
            total_cloudy += raw_per_day[is_cloudy_day].sum().item()
            n_cloudy     += int(is_cloudy_day.sum().item())
            total_cloudy_days += int((regime_ids_d != 0).sum().item())
            total_days        += B * W

        if train:
            optim.zero_grad()
            loss.backward()
            gnorm = _compute_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if ema is not None:
                ema.update(model)
            total_gnorm += gnorm

        total_loss += loss.item()
        total_l_temporal += float(l_temp.item())
        total_l_variance += float(l_var.item())
        n_batches  += 1

        # LEAK 1: free large GPU tensors explicitly after each batch.
        # v_pred holds the autograd graph until Python GC runs (next iteration).
        # Explicitly deleting after backward() and optim.step() ensures
        # CUDA can reuse this memory for the next batch's forward pass,
        # preventing peak memory from being 2x the steady-state usage.
        del v_pred, target, tau_k, noise, tau0, z
        if tau0_hat_for_aux is not None:
            del tau0_hat_for_aux
            tau0_hat_for_aux = None

        if train and log_every > 0 and n_batches % log_every == 0:
            log.debug(
                "  diff step=%d  loss=%.4f  snr_mean=%.2f  gnorm=%.3f  inpaint=%s",
                n_batches, loss.item(), float(_snr_mean),
                total_gnorm / max(n_batches, 1),
                "Y" if is_inpainting else "N",
            )

    nb = max(n_batches, 1)
    _snr_std = (_snr_M2 / max(_snr_n - 1, 1)) ** 0.5 if _snr_n > 1 else 0.0
    return {
        "loss":            total_loss   / nb,
        "loss_clear":      total_clear  / max(n_clear,  1),
        "loss_cloudy":     total_cloudy / max(n_cloudy, 1),
        "n_clear_days":    float(n_clear),
        "n_cloudy_days":   float(n_cloudy),
        "snr_mean":        float(_snr_mean),
        "snr_std":         float(_snr_std),
        "grad_norm":       total_gnorm  / nb,
        "tau_mean":        total_tau_mean / nb,
        "tau_std":         total_tau_std  / nb,
        "n_tau_overflow":  float(n_tau_overflow),
        "inpaint_loss":    total_inp_loss / max(n_inp, 1),
        "n_inpaint_steps": float(n_inp),
        "obs_cloudy_frac": total_cloudy_days / max(total_days, 1),
        "l_temporal":      total_l_temporal / nb,
        "l_variance":      total_l_variance / nb,
    }


def _log_per_layer_grads_from_loader_diff(
    model: SolarDenoiser,
    vae: SolarVAE,
    loader: DataLoader,
    optim,
    schedule: TauNoiseSchedule,
    latent_transform: LatentTauTransform,
    device: torch.device,
    cfg: Dict,
    epoch: int,
) -> None:
    dc = cfg["diffusion"]
    model.train()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    B, W, T_max = batch["k_star"].shape
    _phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))  # C7
    with torch.no_grad():
        k_star  = batch["k_star"].to(device).reshape(B * W, T_max)
        phys_2d = batch["intraday_phys"].to(device).reshape(B * W, T_max, _phys_dim)
        mask_2d = batch["valid_mask"].to(device).reshape(B * W, T_max)
        obs_2d  = _get_obs_phys(batch, device, B, W, T_max)
        z, _ = vae.encode(k_star, phys_2d, mask_2d, obs_2d)
        z     = z.reshape(B, W, -1)
        tau0 = latent_transform.to_tau_space(z)
    if bool(dc.get("per_day_k_sampling", True)):
        k_step = torch.randint(0, dc["num_steps"], (B, W), device=device)
    else:
        k_step = torch.randint(0, dc["num_steps"], (B,), device=device)
    noise        = torch.randn_like(tau0)
    tau_k, noise = schedule.q_sample(tau0, k_step, noise)
    day_feat   = batch["day_features"].to(device)
    location   = batch["location"].to(device)
    intra_phys = batch["intraday_phys"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    drop_mask  = torch.zeros(B, device=device, dtype=torch.bool)
    v_pred = model(tau_k, k_step, day_feat, location,
                   intra_phys, valid_mask, drop_mask=drop_mask,
                   regime_ids=batch["regime"].to(device))
    pred_type = dc["prediction_type"]
    if pred_type == "v":
        target = schedule.get_v_target(tau0, noise, k_step)
    elif pred_type == "eps":
        target = noise
    else:
        target = tau0
    if k_step.dim() == 2:
        ab_k  = schedule.alpha_bars[k_step.reshape(-1).long()].reshape(k_step.shape)
    else:
        ab_k  = schedule.alpha_bars[k_step.long()]
    snr_k = ab_k / (1.0 - ab_k + 1e-8)
    loss, _, _ = diffusion_loss(v_pred, target, snr=snr_k, min_snr_gamma=dc["min_snr_gamma"])
    # Note 3 FIX: use try/finally so model.eval() is always restored even if
    # backward() or _log_per_layer_grads raises. Previously an exception between
    # backward() and model.eval() would leave the model in train() mode for the
    # remainder of the epoch, corrupting all subsequent val passes.
    optim.zero_grad()   # clear any stale grads before backward
    try:
        loss.backward()
        _log_per_layer_grads(model, epoch + 1, "Diffusion")
    finally:
        optim.zero_grad()   # discard diagnostic grads - do NOT step
        model.eval()


# =============================================================================
# Stage 2 - main training loop
# =============================================================================

def train_diffusion(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vae: SolarVAE,
    latent_transform: LatentTauTransform,
) -> SolarDenoiser:
    #Train the tau-space DDPM denoiser.
    device = get_device(cfg)
    tc     = cfg["training"]["diffusion"]
    dc     = cfg["diffusion"]

    # -- Startup double-compensation check ------------------------------------
    # WeightedRandomSampler (main.py) already rebalances to ~equal regime fractions.
    # A high max weight in regime_loss_weights_diff on top of that over-corrects
    # and biases generation toward overcast, collapsing P(C->C).
    # Warn here (before training starts) rather than only per-epoch.
    _n_reg_diff = int(dc.get("n_regimes", 4))
    _n_comp_gmm = int(cfg["data"]["regime"].get("n_components", 4))
    if _n_reg_diff != _n_comp_gmm:
        raise ValueError(
            f"[STARTUP] diffusion.n_regimes={_n_reg_diff} != "
            f"data.regime.n_components={_n_comp_gmm}. These must be identical."
        )
    _diff_weights = dc.get("regime_loss_weights_diff", [])
    if len(_diff_weights) != _n_reg_diff:
        raise ValueError(
            f"[STARTUP] diffusion.regime_loss_weights_diff has {len(_diff_weights)} entries "
            f"but diffusion.n_regimes={_n_reg_diff}. They must match exactly."
        )
    _max_w = max(_diff_weights)
    _startup_w_warn  = float(cfg["diagnostics"].get("diff_startup_weight_warn", 2.5))
    _use_sampler_diff = bool(cfg["training"]["diffusion"].get("use_sampler", False))
    if _use_sampler_diff and _max_w > _startup_w_warn:
        log.warning(
            "[STARTUP] max regime_loss_weights_diff=%.1f > %.1f with WeightedRandomSampler active. "
            "The sampler already balances regimes. "
            "Using a high per-class weight on top will over-represent cloudy days "
            "and collapse P(clear->clear) in generated sequences. "
            "Recommended: max weight <= %.1f when using the sampler.",
            _max_w, _startup_w_warn, _startup_w_warn,
        )

    schedule = TauNoiseSchedule(
        num_steps=dc["num_steps"],
        alpha_bar_min=dc["alpha_bar_min"],
        schedule=dc["noise_schedule"],
        cosine_s=float(dc.get("cosine_s", 0.008)),
        device=device,
    )

    # C6: Validate that loaded tau-transform affine stats match the current
    # encoder output distribution. If z_std drifted during VAE training, tau_mu
    # and tau_sig become stale and cause SNR miscalibration at every diffusion step.
    # We sample a small batch from the train_loader to check the live distribution.
    try:
        _c6_batch = next(iter(train_loader))
        _c6_B, _c6_W, _c6_T = _c6_batch["k_star"].shape
        _c6_phys_dim = int(cfg["vae"].get("intraday_physics_dim", 3))
        with torch.no_grad():
            _c6_k  = _c6_batch["k_star"].to(device).reshape(_c6_B * _c6_W, _c6_T)
            _c6_ph = _c6_batch["intraday_phys"].to(device).reshape(_c6_B * _c6_W, _c6_T, _c6_phys_dim)
            _c6_ma = _c6_batch["valid_mask"].to(device).reshape(_c6_B * _c6_W, _c6_T)
            _c6_ob = _get_obs_phys(_c6_batch, device, _c6_B, _c6_W, _c6_T)
            _c6_z, _ = vae.encode(_c6_k, _c6_ph, _c6_ma, _c6_ob)
            # C6 bug: pass FULL z_full (116 dims) to validate_against.
            # Previously we sliced to z_flat (112 dims), but validate_against
            # calls to_tau_space() which applies flip_dims mask fitted on 116 dims.
            # Passing 112 dims to a 116-dim mask causes a shape mismatch crash.
            # The transform was fitted on z_full so pass z_full here.
            _c6_z_full = _c6_z.cpu().numpy()  # (N, 116) - full z including z_var
        latent_transform.validate_against(_c6_z_full)
        del _c6_batch, _c6_k, _c6_ph, _c6_ma, _c6_ob, _c6_z, _c6_z_full
    except Exception as _c6_e:
        log.warning("[C6] tau-transform validation skipped: %s", _c6_e)

    # Verify that inpainting context length matches the inference sliding-window
    # overlap.  n_ctx_train must equal W - inference.stride so that inpainting
    # training mirrors the RePaint conditioning used at generation time.
    _W          = cfg["data"]["window_size"]
    _inf_stride = cfg["inference"]["stride"]
    _n_ctx      = int(dc.get("n_ctx_train", 9))
    _expected   = _W - _inf_stride
    if _n_ctx != _expected:
        # Note 7 FIX: raise instead of warn. A mismatch here means inpainting
        # training uses a different context window than RePaint inference, which
        # silently degrades autoregressive generation quality over long sequences.
        # This is always a misconfiguration in production; set
        # diffusion.n_ctx_train = window_size - inference.stride in config.yaml.
        raise ValueError(
            f"[STARTUP] n_ctx_train={_n_ctx} != "
            f"window_size({_W}) - inference.stride({_inf_stride}) = {_expected}. "
            f"Inpainting training context must match RePaint inference overlap. "
            f"Fix: set diffusion.n_ctx_train: {_expected} in config.yaml."
        )

    model = SolarDenoiser(cfg).to(device)
    ema   = EMA(model, decay=tc["ema_decay"])
    optim = AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    warmup = LinearLR(optim, start_factor=tc["warmup_start_factor"], end_factor=1.0,
                      total_iters=tc["warmup_epochs"])
    cosine = CosineAnnealingLR(optim,
                               T_max=max(tc["epochs"] - tc["warmup_epochs"], 1),
                               eta_min=tc["lr_min"])
    sched = SequentialLR(optim, schedulers=[warmup, cosine],
                         milestones=[tc["warmup_epochs"]])

    best_val_raw  = float("inf")
    ema_val       = None
    best_ema_val  = float("inf")
    ema_alpha     = tc["val_ema_alpha"]
    best_path     = Path(cfg["paths"]["diffusion_checkpoint"])
    # Periodic checkpoint - overwritten every save_every_n_epochs epochs.
    _diff_save_every    = int(cfg.get("logging", {}).get("save_every_n_epochs", 20))
    periodic_path_diff  = best_path.parent / (best_path.stem + "_periodic" + best_path.suffix)
    patience      = tc["early_stopping_patience"]
    epochs_no_imp = 0
    log_every     = tc["log_every_n_steps"]
    grad_log_freq = tc["log_grad_every_n_epochs"]

    diag_cfg      = cfg.get("diagnostics", {})
    tau_diag_freq = diag_cfg.get("tau_diag_freq",  5)
    gen_diag_freq = diag_cfg.get("gen_diag_freq",  5)   # replaces slow trans_diag_freq
    # MEMORY: default gen_diag_batches lowered to 4.
    # Each batch runs 20 DDIM steps + 672 individual VAE decodes.
    # 8 batches was causing OOM on smaller GPUs. 4 is statistically sufficient.
    gen_diag_batches = diag_cfg.get("gen_diag_batches", 4)  # val batches for fast diag

    best_diag: Dict[str, float] = {}
    last_gen_diag: Optional[Dict] = None
    best_train_metrics: Dict[str, float] = {}

    # Resume from checkpoint if one exists, restoring EMA shadow so smoothing
    # history is preserved (shadow != model weights after a proper save).
    start_epoch = 0
    if best_path.exists():
        ckpt = load_checkpoint(best_path, model, optim)
        start_epoch = ckpt.get("epoch", 0) + 1
        if "ema_shadow" in ckpt:
            # C4: restore EMA shadow into both the ema object AND the model.
            # The saved metric (best_ema_val) was computed on EMA weights. Loading
            # raw model_state means the resumed model starts ~38% worse than reported.
            # Solution: copy ema_shadow into ema.shadow (restores smoothing history)
            # AND into model parameters (so the first val pass matches best_ema_val).
            for k, v in ema.shadow.items():
                if k in ckpt["ema_shadow"]:
                    v.data.copy_(ckpt["ema_shadow"][k].to(v.device))
            # Apply EMA weights to model so training resumes from the best smoothed state
            ema.copy_to(model)
            log.info(
                "EMA shadow restored and applied to model (epoch=%d). "
                "Model now starts from EMA weights, not raw checkpoint weights.",
                start_epoch - 1,
            )
        else:
            log.warning(
                "No ema_shadow in checkpoint; EMA history reset to model weights. "
                "First val pass may show higher loss than saved metric."
            )
        best_ema_val  = ckpt.get("metric", float("inf"))
        best_val_raw  = ckpt.get("best_val_raw", float("inf"))
        epochs_no_imp = ckpt.get("epochs_no_imp", 0)
        # Also check the sidecar resume-state file for more up-to-date patience
        # state (written every epoch, not just on best).
        _resume_state_path = best_path.with_suffix(".resume_state.json")
        if _resume_state_path.exists():
            try:
                import json as _json
                _rs = _json.loads(_resume_state_path.read_text())
                best_val_raw  = float(_rs.get("best_val_raw",  best_val_raw))
                epochs_no_imp = int(  _rs.get("epochs_no_imp", epochs_no_imp))
                log.info("Resume state loaded: best_val_raw=%.5f  epochs_no_imp=%d",
                         best_val_raw, epochs_no_imp)
            except Exception as _e:
                log.warning("Could not read resume state file: %s", _e)
        for _ in range(max(start_epoch - 1, 0)):   # SequentialLR.__init__ already called step() once
            sched.step()
        log.info("Resumed diffusion training from epoch %d  (best_ema=%.5f  no_imp=%d)",
                 start_epoch, best_ema_val, epochs_no_imp)

    # JSON training log - one record per epoch, append-only for safe resume.
    diff_jsonlog = Path(cfg.get("paths", {}).get(
        "diffusion_training_log", "logs/diffusion_training_log.jsonl"))
    log.info("  JSON epoch log -> %s", diff_jsonlog)

    log.info("=== Stage 2: Diffusion training for %d epochs (patience=%d) ===",
             tc["epochs"], patience)
    log.info(
        "  lr=%.2e  lr_min=%.2e  warmup=%d  grad_clip=%.1f  "
        "ema_decay=%.4f  val_ema_alpha=%.2f",
        tc["lr"], tc["lr_min"], tc["warmup_epochs"], tc["grad_clip"],
        tc["ema_decay"], ema_alpha,
    )
    log.info(
        "  pred=%s  K=%d  schedule=%s  min_snr_gamma=%.1f  "
        "regime_loss_weights=%s  cfg_drop=%.2f",
        dc["prediction_type"], dc["num_steps"], dc["noise_schedule"],
        dc["min_snr_gamma"], dc["regime_loss_weights_diff"], dc["cfg_dropout_prob"],
    )
    # Warn when per-class loss weights are high alongside active regime resampling.
    # WeightedRandomSampler already balances batches to ~equal regime fractions.
    # Stacking a high loss weight on top causes double-compensation (Case 1):
    # overcast is both oversampled AND given a higher per-sample loss weight,
    # which biases generation toward overcast and collapses P(C->C).
    _max_w = max(dc["regime_loss_weights_diff"])
    _epoch_w_warn     = float(cfg["diagnostics"].get("diff_epoch_weight_warn", 3.0))
    _use_sampler_diff = bool(cfg["training"]["diffusion"].get("use_sampler", False))
    if _use_sampler_diff and _max_w > _epoch_w_warn:
        log.warning(
            "  [!] max regime_loss_weights_diff=%.1f is high. WeightedRandomSampler "
            "already balances regimes; a high per-class loss weight stacks on top and "
            "over-represents cloudy days (double-compensation). "
            "Recommend max weight <= %.1f when resampling is enabled.",
            _max_w, float(cfg["diagnostics"].get("diff_startup_weight_warn", 2.5)),
        )
    log.info("  Model parameters: %d", sum(p.numel() for p in model.parameters()))
    log.info(
        "  Diagnostics: tau_diag_freq=%d  gen_diag_freq=%d  gen_diag_batches=%d",
        tau_diag_freq, gen_diag_freq, gen_diag_batches,
    )

    _log_regime_frequencies(val_loader, cfg, label="Val")

    # -- Adaptive training controller ------------------------------------------
    _diff_controller = TrainingController(cfg, stage="diff")

    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        train_m = _run_diff_epoch(
            model, vae, train_loader, optim, schedule, latent_transform,
            device, cfg, tc["grad_clip"], log_every, train=True, ema=ema,
        )

        model.eval()
        with torch.no_grad():
            val_m = _run_diff_epoch(
                model, vae, val_loader, None, schedule, latent_transform,
                device, cfg, tc["grad_clip"], log_every, train=False,
            )

        sched.step()

        if ema_val is None:
            ema_val = val_m["loss"]
        else:
            ema_val = (1.0 - ema_alpha) * ema_val + ema_alpha * val_m["loss"]

        param_norm = _compute_param_norm(model)
        ema_diff   = _compute_ema_param_diff(model, ema)

        is_best_raw = val_m["loss"] < best_val_raw
        if is_best_raw:
            best_val_raw       = val_m["loss"]
            best_train_metrics = dict(train_m)
            # Bug fix: previously we copy_to(model) before saving, so the
            # checkpoint stored EMA weights in model_state and the raw weights
            # were thrown away.  On resume, ema.shadow == model weights, meaning
            # the EMA had lost its history and the smoothing benefit reset to zero.
            # Fix: save raw model_state + ema_shadow separately.  At inference,
            # load the ema_shadow into the model explicitly (see load_checkpoint).
            save_checkpoint(best_path, model, optim, epoch, val_m["loss"],
                            {"ema_shadow": {k: v.clone() for k, v in ema.shadow.items()}})

        # C3: patience now tracks RAW val loss improvement, not EMA.
        # Checkpoint is saved on raw val loss -> patience must match the same criterion.
        # Previously: EMA patience could fire while raw val still improved, causing
        # premature stopping (EMA takes ~1/alpha epochs to reflect a genuine gain).
        # EMA is still updated and logged as a smoothing indicator, not a gate.
        is_ema_better = ema_val < best_ema_val  # tracked for logging only
        if is_ema_better:
            best_ema_val = ema_val
        if is_best_raw:
            # Raw val improved -> reset patience (consistent with checkpoint criterion)
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        # Periodic diffusion checkpoint - overwritten every _diff_save_every epochs.
        if (epoch + 1) % _diff_save_every == 0:
            save_checkpoint(periodic_path_diff, model, optim, epoch, val_m["loss"],
                            {"ema_shadow": {k: v.clone() for k, v in ema.shadow.items()},
                             "periodic": True,
                             "best_val_raw": best_val_raw,
                             "epochs_no_imp": epochs_no_imp})
            log.info("  [PERIODIC CKPT] diffusion epoch=%d saved -> %s",
                     epoch + 1, periodic_path_diff)

        lc_val           = train_m["loss_clear"]
        ratio_meaningful = train_m.get("n_clear_days", 0.0) > 0 and lc_val > 1e-4
        ratio_now        = train_m["loss_cloudy"] / lc_val if ratio_meaningful else float("nan")

        log.info(
            "Epoch %3d/%d | "
            "train [loss=%.5f  clear=%.5f  cloudy=%.5f  "
            "ratio=%s  snr=%.2f  gnorm=%.3f] | "
            "val [loss=%.5f  ema=%.5f] | "
            "lr=%.2e  pnorm=%.3f  ema_diff=%.4f  no_imp=%d%s",
            epoch + 1, tc["epochs"],
            train_m["loss"],
            train_m["loss_clear"],
            train_m["loss_cloudy"],
            f"{ratio_now:.2f}" if ratio_meaningful else "n/a",
            train_m["snr_mean"],
            train_m["grad_norm"],
            val_m["loss"],
            ema_val,
            optim.param_groups[0]["lr"],
            param_norm,
            ema_diff,
            epochs_no_imp,
            "  BEST" if is_best_raw else "",
        )
        log.info(
            "         | tau health: mean=%.3f  std=%.3f  overflow_batches=%.0f"
            "  snr_std=%.2f  inpaint_loss=%.4f(n=%.0f)  obs_cloudy=%.1f%%  per_day_k=%s",
            train_m["tau_mean"], train_m["tau_std"],
            train_m["n_tau_overflow"], train_m["snr_std"],
            train_m.get("inpaint_loss", 0.0), train_m.get("n_inpaint_steps", 0.0),
            100.0 * train_m.get("obs_cloudy_frac", 0.0),
            "Y" if dc.get("per_day_k_sampling", True) else "N",
        )

        # -- per-epoch failure checks Stage 2 ---------------------------------
        ratio_err_thr  = float(cfg["diagnostics"].get("cloudy_clear_ratio_err",  6.0))
        ratio_warn_thr = float(cfg["diagnostics"].get("cloudy_clear_ratio_warn", 4.0))
        tau_mean_warn  = float(cfg["diagnostics"].get("tau_mean_warn", 1.0))
        tau_std_lo     = float(cfg["diagnostics"].get("tau_std_healthy_lo", 0.3))
        tau_std_hi     = float(cfg["diagnostics"].get("tau_std_healthy_hi", 2.5))
        gnorm_warn     = float(cfg["diagnostics"].get("grad_norm_warn", 10.0))
        dc_obs_cf_warn = float(cfg["diagnostics"].get("double_comp_obs_cloudy_warn", 0.55))
        dc_w_warn      = float(cfg["diagnostics"].get("double_comp_weight_warn", 1.5))

        if ratio_meaningful:
            if ratio_now > ratio_err_thr:
                log.warning(
                    "  [!!] cloudy/clear loss ratio=%.2f - model not learning cloudy regimes. "
                    "Raise regime_loss_weights_diff.",
                    ratio_now,
                )
            elif ratio_now > ratio_warn_thr:
                log.warning(
                    "  [!] cloudy/clear loss ratio=%.2f > %.1f - consider raising "
                    "regime_loss_weights_diff.",
                    ratio_now, ratio_warn_thr,
                )
        obs_cf     = train_m.get("obs_cloudy_frac", 0.0)
        max_diff_w = max(dc["regime_loss_weights_diff"])
        if _use_sampler_diff and obs_cf > dc_obs_cf_warn and max_diff_w > dc_w_warn:
            log.warning(
                "  [!] obs_cloudy_frac=%.1f%% AND max regime_loss_weight_diff=%.1f - "
                "possible double-compensation with WeightedRandomSampler.",
                100.0 * obs_cf, max_diff_w,
            )
        if train_m["n_tau_overflow"] > 0:
            log.warning(
                "  [!] tau overflow in %.0f batches - LatentTauTransform needs re-fitting.",
                train_m["n_tau_overflow"],
            )
        if abs(train_m["tau_mean"]) > tau_mean_warn:
            log.warning("  [!] tau_mean=%.3f - tau-transform bias. Re-fit.", train_m["tau_mean"])
        if train_m["tau_std"] > tau_std_hi or train_m["tau_std"] < tau_std_lo:
            log.warning(
                "  [!] tau_std=%.3f outside [%.1f, %.1f] - re-fit LatentTauTransform.",
                train_m["tau_std"], tau_std_lo, tau_std_hi,
            )
        if train_m["grad_norm"] > gnorm_warn:
            log.warning("  [!] gnorm=%.3f > %.1f - tighten grad_clip.", train_m["grad_norm"], gnorm_warn)

        if (epoch + 1) % tau_diag_freq == 0:
            with torch.no_grad():
                diag = _diff_val_diagnostics(
                    model, vae, val_loader, schedule, latent_transform, device, cfg)
            torch.cuda.empty_cache()
            _log_diff_diagnostics(diag, cfg)
            if is_best_raw:
                best_diag = diag

        if (epoch + 1) % gen_diag_freq == 0:
            with torch.no_grad():
                gen_d = _fast_generation_diagnostics(
                    model, vae, val_loader, schedule, latent_transform,
                    device, cfg, n_batches=gen_diag_batches,
                )
            torch.cuda.empty_cache()
            _log_gen_diagnostics(gen_d, cfg)
            last_gen_diag = gen_d

        # -- Adaptive controller -----------------------------------------------
        # Must be called after all metrics are computed.  tc["grad_clip"] is
        # re-read from cfg at the start of each _run_diff_epoch call so any
        # tightening takes effect immediately on the next epoch.
        _ctrl_actions_diff = _diff_controller.step_diff(
                epoch, train_m, val_m,
                gen_diag=gen_d          if (epoch + 1) % gen_diag_freq == 0 else None,
                tau_diag=diag           if (epoch + 1) % tau_diag_freq == 0 else None,
            )

        # WARN 1 FIX: apply nan_guard LR change to the live optimizer.
        # The controller mutates cfg["training"]["diffusion"]["lr"] but AdamW
        # never re-reads cfg after construction - the change was silently ignored.
        for _a in _ctrl_actions_diff:
            if _a.rule == "nan_guard":
                for _pg in optim.param_groups:
                    _pg["lr"] = _a.new_val
                log.info("[CTRL/diff] nan_guard: patched optimizer LR -> %.3e", _a.new_val)

        # -- JSON epoch record -------------------------------------------------
        diff_record: Dict = {
            "stage": "diffusion",
            "epoch": epoch + 1,
            "is_best_raw": bool(is_best_raw),
            "is_ema_better": bool(is_ema_better),
            "lr": float(optim.param_groups[0]["lr"]),
            # live controller parameter snapshot
            "ctrl_grad_clip":         float(tc["grad_clip"]),
            "ctrl_diff_weights":      list(cfg["diffusion"].get("regime_loss_weights_diff", [])),
            # train metrics
            "train_loss":          train_m["loss"],
            "train_loss_clear":    train_m["loss_clear"],
            "train_loss_cloudy":   train_m["loss_cloudy"],
            "train_n_clear_days":  train_m["n_clear_days"],
            "train_n_cloudy_days": train_m["n_cloudy_days"],
            "train_cloudy_clear_ratio": float(ratio_now) if ratio_meaningful else float("nan"),
            "train_snr_mean":      train_m["snr_mean"],
            "train_snr_std":       train_m["snr_std"],
            "train_grad_norm":     train_m["grad_norm"],
            "train_tau_mean":      train_m["tau_mean"],
            "train_tau_std":       train_m["tau_std"],
            "train_n_tau_overflow": train_m["n_tau_overflow"],
            "train_inpaint_loss":  train_m.get("inpaint_loss", float("nan")),
            "train_n_inpaint_steps": train_m.get("n_inpaint_steps", float("nan")),
            "train_obs_cloudy_frac": train_m.get("obs_cloudy_frac", float("nan")),
            # val metrics
            "val_loss":            val_m["loss"],
            "val_loss_clear":      val_m["loss_clear"],
            "val_loss_cloudy":     val_m["loss_cloudy"],
            "val_ema_loss":        float(ema_val) if ema_val is not None else float("nan"),
            "ema_param_diff":      float(ema_diff),
            "param_norm":          float(param_norm),
            "epochs_no_imp":       epochs_no_imp,
        }
        # Attach tau diag block if computed this epoch
        if (epoch + 1) % tau_diag_freq == 0:
           _rnames = cfg["data"]["regime"].get(
               "class_names", ["clear", "mixed_clear", "mixed_overcast", "overcast"]
           )[:int(cfg["diffusion"].get("n_regimes", 4))]
           _regime_loss_block = {
               f"diag_loss_{nm}": diag.get(f"loss_{nm}", float("nan"))
               for nm in _rnames
           }
           diff_record.update(_regime_loss_block)
           diff_record.update({
               "diag_loss_snr_low":    diag.get("loss_snr_low",  float("nan")),
               "diag_loss_snr_mid":    diag.get("loss_snr_mid",  float("nan")),
               "diag_loss_snr_high":   diag.get("loss_snr_high", float("nan")),
               "diag_tau_mean":        diag.get("tau_mean",         float("nan")),
               "diag_tau_std":         diag.get("tau_std",          float("nan")),
               "diag_tau_clear_mean":  diag.get("tau_clear_mean",   float("nan")),
               "diag_tau_cloudy_mean": diag.get("tau_cloudy_mean",  float("nan")),
               "diag_tau_separation":  diag.get("tau_separation",   float("nan")),
           })
        # Attach gen diag block if computed this epoch
        if (epoch + 1) % gen_diag_freq == 0:
            diff_record.update({
                "gen_diversity_ratio":   gen_d.get("diversity_ratio",   float("nan")),
                "gen_window_var_ratio":  gen_d.get("window_var_ratio",  float("nan")),
                "gen_coverage_dark_gap": gen_d.get("coverage_dark_gap", float("nan")),
                "gen_coverage_brt_gap":  gen_d.get("coverage_brt_gap",  float("nan")),
                "gen_real_dark":         gen_d.get("real_dark",         float("nan")),
                "gen_gen_dark":          gen_d.get("gen_dark",          float("nan")),
                "gen_real_brt":          gen_d.get("real_brt",          float("nan")),
                "gen_gen_brt":           gen_d.get("gen_brt",           float("nan")),
                "gen_real_wvar":         gen_d.get("real_wvar",         float("nan")),
                "gen_gen_wvar":          gen_d.get("gen_wvar",          float("nan")),
            })
        diff_record.update(ctrl_actions_to_record(_ctrl_actions_diff))
        _append_json_log(diff_jsonlog, diff_record)

        # Write a small resume-state file each epoch so that best_val_raw and
        # epochs_no_imp survive a crash/restart.  Overwritten every epoch.
        _resume_state_path = best_path.with_suffix(".resume_state.json")
        try:
            import json as _json
            _resume_state_path.write_text(_json.dumps({
                "epoch":         epoch,
                "best_val_raw":  float(best_val_raw),
                "epochs_no_imp": epochs_no_imp,
            }))
        except Exception as _e:
            log.warning("Could not write resume state: %s", _e)

        if epochs_no_imp >= patience:
            log.info("Early stopping at epoch %d (patience=%d).", epoch + 1, patience)
            break

    ckpt = load_checkpoint(best_path, model)
    # Apply EMA shadow weights to get the smoothed final model.
    # The checkpoint stores raw model_state + ema_shadow separately so
    # resume correctly restores EMA history (shadow != model weights).
    if "ema_shadow" in ckpt:
        log.info("Applying EMA shadow weights to final model.")
        for k, v in model.named_parameters():
            if k in ckpt["ema_shadow"]:
                v.data.copy_(ckpt["ema_shadow"][k])

    if not best_diag:
        with torch.no_grad():
            best_diag = _diff_val_diagnostics(
                model, vae, val_loader, schedule, latent_transform, device, cfg)
    if last_gen_diag is None:
        with torch.no_grad():
            last_gen_diag = _fast_generation_diagnostics(
                model, vae, val_loader, schedule, latent_transform,
                device, cfg, n_batches=gen_diag_batches,
            )
        _log_gen_diagnostics(last_gen_diag, cfg)

    _diff_action_guide(best_diag, last_gen_diag, best_train_metrics, cfg)
    log.info("Diffusion training complete. Best raw val loss: %.5f", best_val_raw)
    return model