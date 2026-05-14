# Changes since `9e0869c` / `74cb626`

This document records every meaningful change made to `CPH_HousingPriceModel.ipynb` and supporting data files since commits `9e0869c1` (*changes for module 4*) and `74cb626e` (*variational inference adding*), including all subsequent commits and the current uncommitted working-tree changes.

---

## NEW CHANGES

### 1. `year_std_std` removed from the feature set entirely

**Affected files:** `CPH_HousingPriceModel.ipynb` (feature engineering cells 41, 43, 45, 46 and `FEATURE_COLS_STD` in every model cell), `data/cleaned/model_panel_features.csv`, `data/cleaned/model_panel_ungrouped_restricted_features.csv`.

K (number of features) drops from 6 to 5.

**Why:** The model is fitted on *first differences* of log prices. Differencing already removes the long-run price trend. Including a linearly growing time index `year_std` on top of differenced data re-introduces that trend through the learned `beta_year` coefficient.

When forecasting, `year_std` either keeps growing (explosive) or is held at the 2023 value (~+1.63 std, well above the training mean of 0). In either case, `beta_year × year_std > 0` injects a positive offset at every simulation step. Via `cumsum` over 37 steps:

```text
cumulative z-drift ≈ 37 × beta_year × year_std / (1 - phi) >> 0
```

Even over the 5-year test window this was enough to push ARX forecasts ~70% above observed prices.

`phi` already captures temporal persistence. Population, income, and the other demographic features carry the long-run signal.

---

### 2. `mu_phi_raw` prior tightened in Model 5 (no-SEM and SEM)

**Change:** `mu_phi_raw ~ Normal(0.0, 1.0)` → `Normal(0.0, 0.5)`.

**Why:** `tanh(Normal(0, 1))` has most mass in `(−0.96, +0.96)`. An AR(1) coefficient of 0.96 over 37 steps is a near-random walk with no mean reversion. Tightening to `Normal(0, 0.5)` limits the effective range to `tanh(±1) ≈ ±0.76`, a much more stationary regime. `tau_phi ~ HalfNormal(1.0)` (district-to-district heterogeneity) is unchanged.

---

### 3. ARX(2) phi priors made asymmetric: phi1 = 0.5, phi2 = 0.3

**Affected:** Model 3 (ARX(2)), Model 4 (ARX(2)+MLP), and the Model 4 BNN variant.

**Change:** Both priors were `Normal(0, 0.3)` (from commit `e0cb91d`). Now `phi1_raw ~ Normal(0, 0.5)`, `phi2_raw ~ Normal(0, 0.3)`.

**Why:** Lag-1 is the dominant AR coefficient; a wider prior allows it to capture strong price momentum. Lag-2 is secondary and more likely to capture noise on a 16-year window; the tighter prior regularises it. Asymmetric priors reflect the belief that short-lag autocorrelation is more important than long-lag.

---

### 4. Phase 5 — Long-range forecast 2024–2060 added

A new section (5.1–5.5) appended after Model 5 produces district-level DKK/m² forecasts from 2024 to 2060 using the Model 5 NUTS posterior.

**5.1 — Feature projection (KKFR2026)**
`log_total_population_std` follows the official Statistics Denmark cohort projections (KKFR2026) for all 37 forecast years. Amager Vest + Amager Øst are summed to match the nine-district structure. Standardisation uses the raw `StandardScaler` moments from Phase 2 feature engineering. All other features are set to 0 — see change 6.

**5.2 — Posterior predictive simulation**
600 NUTS posterior samples of `(phi_d, beta, sigma_g)` are propagated forward over 37 steps per district. Starting point is the last centered training diff `g0_c_D[d]`.

**5.3 — Back-transform with sanity check**
A guard asserts that no simulated z-space diff exceeds 5.0 in absolute value before the back-transform runs, catching any residual divergence early. Back-transform: `cumsum` from `y_last_z_D` → undo z-score → `exp` (g_mean is not added back — see change 5).

**5.4 — Per-district plots**
Combined historical + test-period CI + long-range CI per district.

**5.5 — All-districts comparison**
Median trajectories for all nine districts in a single figure.

---

### 5. `g_mean` not added back in Phase 5 back-transform

**Change:** `paths_unc = paths_g + g_mean_D[d_idx]` → `paths_unc = paths_g`.

**Why:** `g_mean` is the mean first-difference over the 2008–2023 training period — approximately +4–5% per year real price growth driven by the Copenhagen housing boom (falling rates, rapid urbanisation). Adding it back to every simulated step assumes that exact growth rate continues indefinitely.

`1.044^37 ≈ 5×`. Starting from 50 000 DKK/m² in 2023, the median forecast would reach ~250 000 DKK/m² — economically implausible as a central projection. The AR coefficient `phi_d` captures price persistence; the population covariate captures the structural demographic component.

*Observed test prices still add `g_mean` back correctly* because `g_test_c_D` are actual historical observations (centered by subtracting `g_mean`) and must be un-centered to recover the real price series.

---

### 6. Non-population features set to 0 (training mean) in Phase 5 feature projection

**Change:** `feat_proj[:, k_idx] = float(val_2023[feat].values[0])` → `feat_proj[:, k_idx] = 0.0` for all features except `log_total_population_std`.

**Why:** Features are globally standardised (mean 0 across all district-years 2008–2023). By 2023, Copenhagen incomes and population densities have grown above the long-run training mean — their standardised values sit at +1 to +1.5 std. If `beta` for those features is positive (learned because high-income years coincided with high price growth), holding them at 2023 levels injects a positive `X @ beta > 0` offset at every step:

```text
stationary mean of g_c = (X_2023 @ beta) / (1 - phi) > 0
cumulative drift over 37 years = 37 × stationary_mean
```

Setting non-population features to 0 neutralises this spurious co-trending. The forecast then answers a specific question: *given expected population change (KKFR2026), and assuming other socioeconomic conditions at their long-run average, how do prices evolve?* Population is the one exception because an authoritative 37-year external projection (KKFR2026) exists for it.

---

### 7. SVI cell save/restore wrapper (Phase 5 variable protection)

**Why:** The SVI methodological comparison cell for Model 5 re-runs `_block_one_district` and redefines `g0_c_D`, `X_g_future_D`, `g_test_c_D`, `phi_d_s`, `beta_s`, `sig_s`, `post_h`, and other variables that Phase 5 reads from the NUTS posterior. Running the SVI cell in any order would silently corrupt Phase 5 results.

**Fix:** A `_phase5_saved` dict captures all Phase 5 state at the top of the SVI cell; a restore block at the bottom reassigns all variables. Phase 5 is now execution-order safe.

---

### 8. Prior/posterior KDE plot added for Model 3 (ARX(2))

A code cell at the end of the Model 3 section plots the prior and posterior distributions of `phi1` and `phi2` on the constrained `tanh` scale. `SD_PHI1_RAW = 0.5` and `SD_PHI2_RAW = 0.3` match the model priors (change 3). The plot uses MCMC posteriors by default; commented-out lines switch to VI posteriors.

---

### 9. Two explanatory markdown notes on ARX upward bias

**Note after Model 2 (ARX(1)) back-transform:**
Explains why the single-district ARX overshoot is a structural limitation, not a code error. The model learns that high-income years are associated with above-average price growth; in the test window, income remains high and the positive `X @ beta` compounds through `cumsum`. Motivates the move to hierarchical Model 5.

**Note after Model 5 test-period plots:**
The same upward bias applies to Model 5's 2019–2023 test-period forecast. Partial pooling regularises `beta` but does not eliminate it. The Phase 5 long-range forecast addresses this by setting non-population features to 0 (change 6).

---

## Earlier committed changes (`9e0869c` → `3eac25f`)

### 10. Model 4 — MLP weight key extraction fix (`9e0869c`)

**What changed:** In the Model 4 simulation loop, posterior MLP weights were retrieved by iterating over `post_dict.values()` and matching by `str.endswith`. Replaced with a one-time key lookup by substring:

```python
# Before
W1 = next(v[s] for k, v in post_dict.items() if k.endswith("hidden.weight"))

# After
key_W1 = next(k for k in post_dict if "hidden.weight" in k)
W1 = post_dict[key_W1][s]
```

**Why:** `endswith` on Pyro's auto-generated parameter names is fragile — the suffix depends on the module registration path and can change silently. Pre-caching the key by substring match is more robust and avoids re-scanning the dict on every sample iteration.

---

### 11. Model 4 — `pyro.module` removed, feature ordering fixed, phi priors tightened (`e0cb91d`)

**a) `pyro.module("mlp", mlp_net)` commented out.**
Using `pyro.module` on an MLP whose weights are also declared via `PyroSample` causes a double-registration conflict: Pyro tries to track the parameters both as a registered module and as latent random variables, leading to incorrect gradient flow. Removing the call leaves `PyroSample` in sole control of the weight distributions.

**b) `feature_cols = sorted(...)` → `feature_cols = FEATURE_COLS_STD`.**
Model 4 was previously sorting features alphabetically, producing a different column order from every other model. This made beta coefficients incomparable across models. Using the shared `FEATURE_COLS_STD` list enforces a consistent ordering.

**c) phi priors tightened: `Normal(0, 0.5)` → `Normal(0, 0.3)` for both `phi1_raw` and `phi2_raw` in Models 3 and 4.**
`tanh(Normal(0, 0.5))` has ~95% of its mass within `tanh(±1) ≈ ±0.76`. Tightening to `Normal(0, 0.3)` pulls the prior closer to zero, regularising against spuriously large lag coefficients on a short (≤16 year) training window. Note: subsequently revised to asymmetric priors (0.5 / 0.3) — see change 3.

---

### 12. Documentation, SEM, and feature-engineering fixes (`e1841e5`)

**a) SEM added to housing aggregation.**
`sem_log_sqm_price` is computed as the standard error of the median log-price per district-year:

```python
housing_agg['sem_log_sqm_price'] = 1.253 * std_log_sqm_price / sqrt(n_transactions)
```

The factor 1.253 converts the standard deviation of a normal distribution to the standard error of its median. Used in Model 5 as per-cell observation noise, allowing the likelihood to down-weight district-years with few transactions and high price variance.

**b) Amager collapse documented.** The join of `Amager Øst` and `Amager Vest` into `Amager` is explicitly noted as a sum (not mean) across all count-type columns. Per-capita features are recomputed as `sum(income) / sum(population)` after the merge.

**c) Feature rationale table added** (markdown). Covers each summary feature, the transformation applied, and the statistical/economic justification (log for multiplicative effects, FTE-weighting for unemployment, etc.).

**d) Miscellaneous markdown fixes** — section numbering, table of contents anchors.

---

### 13. SVI methodological comparisons added for Models 1–4 (`74cb626`)

For each of the four single-district models (AR(1), ARX(1), ARX(2), ARX(2)+MLP), an SVI comparison cell was added immediately after the NUTS inference cell. Each uses:

- **Guide:** `AutoDiagonalNormal` — mean-field variational family, one Gaussian per latent variable.
- **Loss:** `Trace_ELBO` — unbiased single-sample ELBO estimator.
- **Optimiser:** Adam, lr = 0.01, 4 000 steps.
- **Posterior approximation:** `Predictive` with `num_samples = 500`.

**Why SVI as a comparison:** NUTS gives the gold-standard posterior but scales poorly. SVI is faster but makes a mean-field independence assumption. Including both lets the reader see the cost of the variational approximation in posterior quality and uncertainty calibration, and provides groundwork for the hierarchical Model 5 SVI variant where NUTS becomes prohibitively slow.

---

### 14. Model 5 rewritten as hierarchical ARX(1) linear (`d4b3914` / `8b847e5` / `9e1a8e8`)

The original Model 5 (hierarchical ARX(2) + district-specific MLP) was replaced with two cleaner implementations:

**Model 5 — no SEM:** Hierarchical ARX(1) linear across all nine districts. Partial pooling on the AR coefficient: `phi_d = tanh(mu_phi_raw + tau_phi × eps_d)`, shared `beta` vector, shared `sigma_g`.

**Model 5 — with SEM:** Same structure, but the observation likelihood uses per-cell combined noise:

```python
obs_noise = sqrt(sem_obs[d, t]^2 + sigma_g^2)
```

`sigma_g` captures AR process uncertainty; `sem_log_sqm_price` captures measurement noise from finite transaction counts. The model automatically trusts dense-data district-years more.

The old MLP-based Model 5 is kept in the notebook as "Old Model 5" (marked superseded) for reference.

---

### 15. Model 5 — SEM observation noise finalised (`3eac25f`)

`sem_log_sqm_price` was wired end-to-end into the Model 5 NUTS likelihood, with the values flowing correctly from the panel CSV through the data-preparation block into the Pyro `obs` sites.
