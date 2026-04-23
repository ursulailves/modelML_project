# CPH Housing Bayesian Model — Full Project Plan

**Due: 15 May 2026 · Deliverables: self-explanatory notebook + 6-page IEEE report**

---

## Status

| Step | Status |
|---|---|
| Phase 1 — Data ingestion & EDA | ✅ Done |
| 2.1 Aggregate housing prices to district-year | ✅ Done |
| 2.2 Align district names, collapse Amager | ✅ Done |
| 2.3 Join panels → `model_panel_restricted.csv` | ✅ Done |
| 2.4 Reduce covariates to 5–8 features | ⬜ Next |
| 2.5 Feature EDA | ⬜ |
| Phase 3 — PGM design | ⬜ |
| Phase 4 — Pyro implementation & sanity check | ⬜ |
| Phase 5 — Inference (SVI + MCMC) | ⬜ |
| Phase 6 — Results & 2026–2060 forecast | ⬜ |
| Phase 7 — Deliverables | ⬜ |

**Working file:** [EDA_copy.ipynb](../Desktop/DTU_nocloud/spring26/model%20based%20ML/mbml-cph-housing-main/EDA_copy.ipynb)
**Input for all remaining work:** `data/cleaned/model_panel_restricted.csv` (153 rows × ~210 cols, 9 districts, 2008–2024)

---

## Phase 2.4 — Reduce covariates to 5–8 summary features

### Context
`model_panel_restricted` has ~200 wide columns. The Pyro model needs a small, interpretable feature matrix X. All columns are counts/totals derived from Statistics Denmark pivot tables — they must be combined into meaningful per-capita or rate features.

### Target features

| Feature | Source | Derivation |
|---|---|---|
| `total_pop` | `pop_*` cols (KKBEF2) | Sum all `pop_` columns — each cell is a unique (ancestry × age × sex) count, so summing gives total population |
| `share_non_danish` | `pop_*` cols | Sum columns NOT containing `persons_of_danish_origin` / `total_pop` |
| `log_income_per_capita` | `income_*` cols (KKIND3) | Sum `income_disposable_income_*_amount_of_income_dkk_1_000` for men + women → total DKK 1,000; divide by `total_pop`; ×1000 for per-person DKK; take log |
| `unemployment_rate` | `unemp_*` cols (KKLEDIG6) | Sum all `unemp_` columns (total unemployed) / working-age pop (KKBEF2 age 20–59 years) |
| `avg_household_size` | `housing_*` cols (KKHUS1) | Weighted average: `Σ(size_value × count_of_households_of_that_size)` / `total_households` where sizes map `1_person→1`, `2_persons→2`, `3_persons→3`, `4_persons→4`, `5_persons_and_more→5` |
| `year_std` | `year` column | `(year - year.mean()) / year.std()` — standardized linear time covariate |
| `nom_interest_rate_std` *(optional)* | from housing `df` merged by year | National average — include as a direct control, not as the sole time signal |

### Implementation approach

```python
panel = pd.read_csv('data/cleaned/model_panel_restricted.csv')

pop_cols   = [c for c in panel.columns if c.startswith('pop_')]
unemp_cols = [c for c in panel.columns if c.startswith('unemp_')]
inc_cols   = [c for c in panel.columns if c.startswith('income_')]
hh_cols    = [c for c in panel.columns if c.startswith('housing_')]

# Total population
panel['total_pop'] = panel[pop_cols].sum(axis=1)

# Share non-Danish
non_danish_pop_cols = [c for c in pop_cols if 'persons_of_danish_origin' not in c]
panel['share_non_danish'] = panel[non_danish_pop_cols].sum(axis=1) / panel['total_pop']

# Log income per capita
disp_inc_cols = [c for c in inc_cols if 'disposable_income' in c and 'amount_of_income_dkk_1_000' in c]
panel['income_per_capita'] = panel[disp_inc_cols].sum(axis=1) * 1000 / panel['total_pop']
panel['log_income_per_capita'] = np.log(panel['income_per_capita'])

# Unemployment rate (only valid 2008+ — already restricted)
working_age_cols = [c for c in pop_cols if any(g in c for g in ['20_29', '30_39', '40_49', '50_59'])]
panel['working_age_pop'] = panel[working_age_cols].sum(axis=1)
panel['unemployment_rate'] = panel[unemp_cols].sum(axis=1) / panel['working_age_pop']

# Average household size
size_map = [('1_person', 1), ('2_persons', 2), ('3_persons', 3), ('4_persons', 4), ('5_persons_and_more', 5)]
total_hh = pd.Series(0.0, index=panel.index)
weighted  = pd.Series(0.0, index=panel.index)
for slug, val in size_map:
    size_sum = panel[[c for c in hh_cols if c.startswith(f'housing_{slug}')]].sum(axis=1)
    total_hh += size_sum
    weighted  += val * size_sum
panel['avg_household_size'] = weighted / total_hh

# Standardised time
panel['year_std'] = (panel['year'] - panel['year'].mean()) / panel['year'].std()

# Collect final feature matrix columns
FEATURE_COLS = ['total_pop', 'share_non_danish', 'log_income_per_capita',
                'unemployment_rate', 'avg_household_size', 'year_std']

# Standardise all features (z-score) for Pyro — priors on slopes β are O(1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(panel[FEATURE_COLS])
panel[[f'{c}_std' for c in FEATURE_COLS]] = X_scaled

panel.to_csv('data/cleaned/model_panel_features.csv', index=False)
```

**Print diagnostics:** `panel[FEATURE_COLS].describe()`, check for NaN (none expected in restricted panel), check `unemployment_rate` is in range 0–0.3.

---

## Phase 2.5 — Feature EDA

New notebook section `2.4 Feature EDA` after the reduction cell.

1. **Correlation heatmap** — `seaborn.heatmap` of `FEATURE_COLS + ['median_log_sqm_price']`, annotated with Pearson r. Flags collinear pairs (|r| > 0.85 → consider dropping one).
2. **Feature vs price over time per district** — one subplot per feature: x = year, y = feature value, one line per district, coloured consistently. Below each: same x-axis with `median_log_sqm_price`. Lets you visually judge leading/lagging relationships.
3. **Outlier check** — box plots of each feature by district. Flag cells > 3 σ from district mean.
4. **Macro variable check** — scatter `nom_interest_rate` vs `median_log_sqm_price` (should show negative correlation). Confirm the interest rate does NOT proxy for the entire time trend once year is included.

Save: `data/cleaned/model_panel_features.csv` (panel + derived features + standardised features).

---

## Phase 3 — PGM design

### 3.1 Generative model

**Generative process** (write this as a numbered list in the notebook):

```
1. Hyperpriors for district intercepts:
     μ_α  ~ Normal(11.0, 1.0)      # mean log sqm price ≈ exp(11) ≈ 60k DKK/sqm
     σ_α  ~ HalfNormal(0.3)         # between-district spread

2. Covariate slopes (shared, one per feature):
     β_k  ~ Normal(0, 0.5)   for k = 1..K

3. District random intercepts (Plate D, D=9):
     α_d  ~ Normal(μ_α, σ_α)  for d = 1..D

4. Observation model (Plate N, N=153):
     μ_dt = α_d + β^T x̃_dt
     y_dt ~ Normal(μ_dt, σ_obs_dt)

   where:
     x̃_dt   = standardised feature vector (observed, fixed)
     σ_obs_dt = std_log_sqm_price_dt (within-cell price std, observed, fixed)
               — heteroskedastic observation noise, data-informed
```

**Key design choices to explain in notebook:**
- σ_obs_dt is treated as observed (not inferred) — data-informed weights, avoids a global noise assumption
- No prior on x̃_dt (it is observed; putting a prior on an observed childless variable is a common mistake per the course rubric)
- Macro variables (interest rate, inflation) included as observed covariates in x̃, not as the only time signal — demographic features must retain their own direct path to price

### 3.2 Time trend strategy

**Recommended: include `year_std` as one of the K covariates.** This gives a linear time trend with slope β_year. Simpler than AR(1) or GP, avoids temporal dependence structure, and is compatible with forecasting (just plug in future year values).

A district-specific time slope (random slope on year) would be a natural extension but is not required for the course.

### 3.3 Prior justification

Use the [Stan Prior Choice Recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) as reference. Key points to note in the report:
- `Normal(11, 1)` for μ_α: prior probability mass between exp(9)≈8k and exp(13)≈440k DKK/sqm — covers all realistic Copenhagen prices
- `HalfNormal(0.3)` for σ_α: 95% prior mass below 0.6 log-scale units (roughly ×1.8 factor between districts) — realistic given actual price variation
- `Normal(0, 0.5)` for β_k: features are standardised, so β_k = 0.5 means a 1-SD change in a feature → 0.5 log-unit change in price (≈65% increase) — wide but regularising

### 3.4 Plate diagram

Required for the report. Elements:
- **Shaded nodes** (observed): x̃_dt, y_dt, σ_obs_dt
- **Unshaded nodes** (latent): μ_α, σ_α, β_k, α_d
- **Plates**: outer plate D (districts), inner plate N (district-year observations)
- Arrows: μ_α → α_d ← σ_α; α_d + β + x̃_dt → μ_dt → y_dt ← σ_obs_dt

Draw with `daft` (Python) or by hand for the report. Include the numbered generative process alongside.

---

## Phase 4 — Pyro implementation & sanity check

### 4.1 Data preparation

```python
import torch
panel = pd.read_csv('data/cleaned/model_panel_features.csv')

# Encode districts as integer indices 0..8
district_cats = pd.Categorical(panel['district'])
panel['district_idx'] = district_cats.codes
D = panel['district_idx'].nunique()  # 9

FEAT_COLS_STD = [f'{c}_std' for c in FEATURE_COLS]
K = len(FEAT_COLS_STD)

district_idx = torch.tensor(panel['district_idx'].values, dtype=torch.long)
X            = torch.tensor(panel[FEAT_COLS_STD].values,  dtype=torch.float32)
y_obs        = torch.tensor(panel['median_log_sqm_price'].values, dtype=torch.float32)
sigma_obs    = torch.tensor(panel['std_log_sqm_price'].values,   dtype=torch.float32).clamp(min=0.01)
```

### 4.2 Pyro model and guide

```python
import pyro
import pyro.distributions as dist

def model(district_idx, X, sigma_obs, y_obs=None):
    D = int(district_idx.max().item()) + 1
    K = X.shape[1]
    N = X.shape[0]

    mu_alpha    = pyro.sample("mu_alpha",    dist.Normal(11.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(0.3))
    beta        = pyro.sample("beta",        dist.Normal(torch.zeros(K), 0.5 * torch.ones(K)).to_event(1))

    with pyro.plate("districts", D):
        alpha = pyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))

    with pyro.plate("observations", N):
        mu = alpha[district_idx] + (X * beta).sum(-1)
        pyro.sample("y", dist.Normal(mu, sigma_obs), obs=y_obs)

# Use AutoNormal guide for SVI
from pyro.infer.autoguide import AutoNormal
guide = AutoNormal(model)
```

### 4.3 Ancestral sampling sanity check (do before fitting to real data!)

1. Choose known "ground truth" parameters: e.g. `α = [10.5, 10.8, 11.0, 11.2, 11.4, 11.6, 11.7, 11.8, 12.0]` for the 9 districts, `β = [0.3, -0.1, 0.4, -0.2, 0.05, 0.1]`
2. Run `model(district_idx, X, sigma_obs)` without `obs` — samples from the prior predictive
3. Override latent variables with known values using `pyro.poutine.condition` to generate synthetic `y_synthetic`
4. Fit the model on `y_synthetic` using SVI
5. **Check:** posterior credible intervals for each α_d and β_k should contain the true values. If not, the model is misspecified or the guide is too restrictive.

### 4.4 Baseline model

Pooled linear regression: no district random effects, no hierarchical structure. Same β but a single global intercept α.

```python
def baseline_model(district_idx, X, sigma_obs, y_obs=None):
    alpha = pyro.sample("alpha", dist.Normal(11.0, 1.0))
    beta  = pyro.sample("beta",  dist.Normal(torch.zeros(K), 0.5 * torch.ones(K)).to_event(1))
    with pyro.plate("observations", len(X)):
        mu = alpha + (X * beta).sum(-1)
        pyro.sample("y", dist.Normal(mu, sigma_obs), obs=y_obs)
```

Compare ELBO and posterior predictive RMSE to the hierarchical model.

---

## Phase 5 — Inference

### 5.1 SVI with AutoNormal guide

```python
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

losses = []
for step in range(5000):
    loss = svi.step(district_idx, X, sigma_obs, y_obs)
    losses.append(loss)
    if step % 500 == 0:
        print(f"Step {step}: ELBO = {-loss:.2f}")

plt.plot(losses); plt.xlabel("Step"); plt.ylabel("ELBO loss"); plt.title("SVI convergence")
```

**Convergence check:** ELBO should plateau and stop decreasing. If it oscillates, reduce learning rate to 0.001.

### 5.2 Extract SVI posterior samples

```python
from pyro.infer import Predictive

predictive = Predictive(model, guide=guide, num_samples=1000)
svi_samples = predictive(district_idx, X, sigma_obs)
# svi_samples['alpha'] shape: (1000, 9) — posterior samples for each district intercept
# svi_samples['beta']  shape: (1000, K)
```

### 5.3 MCMC NUTS validation (required by rubric)

Run NUTS on the full dataset (153 observations is small enough):

```python
from pyro.infer import MCMC, NUTS

pyro.clear_param_store()
nuts_kernel = NUTS(model, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300, num_chains=1)
mcmc.run(district_idx, X, sigma_obs, y_obs)
mcmc_samples = mcmc.get_samples()
```

**Validation checks:**
- R-hat < 1.01 for all parameters (`pyro.ops.stats.gelman_rubin` or ArviZ)
- Trace plots: chains should mix well, no trends or strong autocorrelation
- Compare posterior means of α_d between SVI and MCMC: should agree within 1 SD
- If SVI and MCMC disagree significantly on β_k, the AutoNormal guide may be too restrictive (consider `AutoMultivariateNormal`)

### 5.4 Posterior predictive check

```python
# Generate posterior predictive samples on training data
ppc_samples = predictive(district_idx, X, sigma_obs)['y']  # (1000, 153)

# Plot: for each district, compare posterior predictive distribution to observed y
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i, (dist_name, ax) in enumerate(zip(district_cats.categories, axes.flatten())):
    mask = panel['district'] == dist_name
    y_true = y_obs[mask].numpy()
    y_pred = ppc_samples[:, mask].numpy()
    ax.plot(panel.loc[mask, 'year'], y_true, 'k.', label='observed')
    ax.plot(panel.loc[mask, 'year'], y_pred.mean(0), 'r-', label='posterior mean')
    ax.fill_between(panel.loc[mask, 'year'],
                    np.percentile(y_pred, 5, axis=0),
                    np.percentile(y_pred, 95, axis=0), alpha=0.3, color='red')
    ax.set_title(dist_name)
```

---

## Phase 6 — Results & 2026–2060 forecast

### 6.1 Posterior coefficient interpretation

```python
# β_k posterior (which features matter?)
fig, axes = plt.subplots(1, K, figsize=(14, 4))
for k, feat_name in enumerate(FEATURE_COLS):
    axes[k].hist(mcmc_samples['beta'][:, k].numpy(), bins=40, density=True)
    axes[k].axvline(0, color='red', linestyle='--')
    axes[k].set_title(feat_name)
plt.suptitle("Posterior distributions of covariate slopes β_k")
```

Key questions to address in the report:
- Does higher income per capita predict higher sqm prices? (Expected: β > 0)
- Does higher unemployment predict lower prices? (Expected: β < 0)
- Does share non-Danish origin predict anything after controlling for income? (Interesting question)
- Do district intercepts α_d rank intuitively? (Expected: Indre By highest, outer districts lower)

### 6.2 Forecast 2026–2060

**Feature projection strategy:**
- `total_pop` + `share_non_danish`: use `future_population_panel.csv` (KKFR2026, projections by age × sex × district)
- `log_income_per_capita` + `unemployment_rate` + `avg_household_size`: no projections exist → use simplest defensible assumption: **extrapolate the trend from 2015–2024** (linear regression per district per feature), or hold constant at 2024 values if no clear trend
- `year_std`: straightforward — just standardize the future year values

```python
future_pop = pd.read_csv('data/cleaned/future_population_panel.csv')
# Derive total_pop, share_non_danish from KKFR2026 projected age × sex counts
# (same logic as 2.4, applied to future_pop columns)

# For other features: fit linear trend per district per feature on 2015–2024
# Extrapolate to 2026–2060

# Assemble future X matrix (350 rows: 9 districts × 35 years), standardise with
# the SAME scaler fitted in 2.4 (important: do not refit scaler on future data)

# Forecast using posterior predictive
future_preds = Predictive(model, posterior_samples=mcmc_samples)(
    future_district_idx, X_future, sigma_obs_future  # use median sigma_obs from training
)['y']  # (500, 350)
```

Plot: one panel per district, x = year (2008–2060), y = log sqm price.
- Training period (2008–2024): observed points + posterior fitted line
- Forecast period (2026–2060): posterior predictive mean + 90% credible band
- Discontinuity at 2025 (no housing data that year) — clearly labelled

### 6.3 Leave-one-year-out validation

Hold out 2023 (last complete year with transactions):

```python
train_mask = panel['year'] < 2023
test_mask  = panel['year'] == 2023

# Fit on train; predict on test
# Compute: RMSE, mean log predictive density (MLPD) vs baseline model
```

Report MLPD: `log p(y_test | x_test, model) = mean over test observations of log(predictive density)`. Higher is better; compare hierarchical vs pooled baseline.

---

## Phase 7 — Deliverables

### 7.1 Notebook final cleanup

Before submission:
1. Restart kernel, run all cells in order — every cell must execute without error
2. Ensure all plots have titles, axis labels, and legible legends
3. Every markdown section explains WHY, not just WHAT
4. Remove any debugging or test cells

### 7.2 IEEE 6-page report structure

| Section | ~Length | Content |
|---|---|---|
| Introduction | 0.5p | Research question, why Copenhagen, what the model adds over hedonic OLS |
| Data | 1p | Housing dataset + Statistics Denmark sources; district map; key descriptive statistics; missingness table |
| PGM | 1.5p | Generative process (numbered list) + plate diagram; prior justification; feature derivation |
| Inference | 0.5p | SVI setup (AutoNormal, Adam, 5000 steps); MCMC NUTS validation; convergence evidence |
| Results | 1.5p | β_k posteriors; district intercepts; posterior predictive check; MLPD vs baseline |
| Forecast + Conclusion | 1p | 2026–2060 fan chart per district; uncertainty discussion; limitations; conclusion |

**Report watchouts:**
- Include plate diagram (required by rubric — graders check this)
- Both VI and MCMC results must appear (required by rubric)
- Discuss prior sensitivity briefly
- Macro variables (interest rate, inflation) must appear as direct controls in the model, not as the sole time signal — mention this design choice explicitly

---

## Critical file paths

| File | Description |
|---|---|
| `EDA_copy.ipynb` | All code lives here |
| `data/cleaned/model_panel_restricted.csv` | Input to Phase 2.4 onwards |
| `data/cleaned/model_panel_features.csv` | Output of Phase 2.4 (feature matrix) |
| `data/cleaned/future_population_panel.csv` | KKFR2026 projections for forecast |
| `data/cleaned/historical_panel_merged_aligned.csv` | Aligned 9-district panel |
