# Physics-Guided Latent Diffusion for Synthetic Solar Irradiance Generation

SOLAD generates arbitrarily long, 10-minute resolution global horizontal irradiance sequences given only a site's latitude, longitude, and date range, without any concurrent meteorological observations.

---

## Key Classes

| Class | File | Description |
|:---|:---|:---|
| `SolarVAE` | `vae.py` | Physics-conditioned 1-D conv autoencoder. Encodes daily clear sky index K\* into `z_flat` + `z_var` via FiLM conditioning on solar geometry. |
| `SolarDenoiser` | `denoiser.py` | Transformer DDPM on windows of day-latents conditioned on location, solar geometry, and four-class sky-regime with CFG. |
| `LatentTauTransform` | `optical_depth.py` | Beer--Lambert reparameterisation K\* = K\_max · exp(−τ). Fit once on the training latent cache. |
| `TauNoiseSchedule` | `optical_depth.py` | Cosine noise schedule with v-prediction, used by training and inference. |
| `generate_sequence` | `generate.py` | Full inference pipeline: solar geometry -> reverse diffusion -> VAE decode -> K\* output. |
| `IntraDayNormStats` | `physics_utils.py` | Loads normalisation stats and assembles intraday physics tensors. |
| `DayFeatureNormStats` | `physics_utils.py` | Loads normalisation stats and assembles day-level feature tensors. |

---

## Installation

```bash
git clone https://github.com/frimane/SOLAD.git
cd SOLAD
pip install -r requirements.txt
```
---

## Running the App

```bash
streamlit run app.py
```

---

## Training Data

This version of the model was trained on [SURFRAD](https://gml.noaa.gov/grad/surfrad/) stations (BON, DRA, FPK, GWN, PSU, SXF, TBL), years 2020–2023, covering a wide range of North American climate regimes. No auxiliary meteorological variables are used -- only solar irradiance records and their solar-geometry-derived features.

---

## Retraining or Fine-Tuning (Optional)

The model is two-stage and trained sequentially. All hyperparameters are in `config.yaml`. Refer to the paper for full architectural and loss details.

### Stage 1 — Train `SolarVAE`

| Step | Action |
|:---|:---|
| 1. Data | Prepare daily K\* profiles at your needed resolution. Compute solar geometry (zenith, ETR, clear-sky GHI) via `pvlib` using `physics_utils.py`. |
| 2. Normalisation | Fit z-score statistics on your training split -> save to `data/norm_intraday.json` and `data/norm_day_feat.json`. |
| 3. Regime labels | Fit a 4-component GMM on per-day K\* statistics. Labels: `0`=clear, `1`=partly-cloudy, `2`=cloudy, `3`=overcast (sorted by descending mean K\*) -> save to `data/regime_gmm.json`. |
| 4. Training | Train `SolarVAE` with the losses in `vae.py`: `reconstruction_loss`, `spectral_loss`, `regime_separation_loss`, `latent_variance_penalty`. |

### Stage 2 — Train `SolarDenoiser`

| Step | Action |
|:---|:---|
| 1. Latent cache | Encode the full training set with frozen `SolarVAE` -> save `z_full`, physics tensors, and regime labels. |
| 2. τ-transform | Fit `LatentTauTransform` on `z_flat = z_full[:, :d_z]` -> save to `inference_bundle/latent_tau_stats.json`. |
| 3. Training | Train `SolarDenoiser` on W-day windows in τ-space using v-prediction and `TauNoiseSchedule`. CFG dropout probability is set in `config.yaml`. |

---

## Citation

```bibtex
@article{frimane2026solad,
  title   = {Physics-Guided Latent Diffusion for Synthetic Solar Irradiance Generation},
  author  = {Frimane, Azeddine and others},
  journal = {under review},
  year    = {2026},
}
```
---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

## Contact

**Azeddine Frimane** — [Azeddine.frimane@yahoo.com](mailto:Azeddine.frimane@yahoo.com)
