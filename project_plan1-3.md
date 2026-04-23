# Plan: Feature Engineering Tasks 1–3 (CPH Housing Bayesian Model)

## Context

The project is a hierarchical Bayesian model of Copenhagen housing prices for a DTU Model-Based ML course. Phases 1–2 (data ingestion and basic EDA) are complete. The next step is to build the joined district-year panel that will feed the Pyro model.

**Current state:**
- `df`: housing DataFrame (150,382 rows) with columns including `neighborhood`, `year`, `sqm_price`, `sales_type`, etc. `neighborhood` uses simplified names from `map_zip_to_district()`.
- `historical_panel`: Statistics Denmark panel, 400 rows × 212 columns, districts 1986–2025, saved at `data/cleaned/historical_panel_merged.csv`. Uses full district names (e.g. `Vesterbro/Kongens Enghave`, `Amager Øst`, `Amager Vest`).
- `data/cleaned/` already contains: `*_tidy.csv` files, `historical_panel_merged.csv`, `future_population_panel.csv`, `combined_panel_with_future_population.csv`.

**Key name mismatches to resolve:**
| `housing_agg` name | `historical_panel` name |
|--------------------|------------------------|
| `Vesterbro` | `Vesterbro/Kongens Enghave` |
| `Amager` | zip 2300 covers both Amager Øst and Amager Vest → collapse both panel rows into one `Amager` row (sum counts) before joining |
| `Frederiksberg` | not in Statistics Denmark tables → drop entirely |
| `Brønshøj-Husum` | same string, but verify UTF-8 encoding |

After combining and joining, the model will have **9 districts**: Indre By, Østerbro, Nørrebro, Vesterbro/Kongens Enghave→Vesterbro, Valby, Vanløse, Brønshøj-Husum, Bispebjerg, Amager.

---

## Implementation approach

All work goes into [EDA_copy.ipynb](EDA_copy.ipynb). The notebook needs two things:
1. **Structural cleanup** — insert markdown section headers so the notebook reads as a coherent Phase 1 → Phase 2 document
2. **New code cells** — Tasks 1–3 appended after the existing Phase 1 content

Critical files:
- [EDA_copy.ipynb](EDA_copy.ipynb) — all work goes here
- [data/cleaned/historical_panel_merged.csv](data/cleaned/historical_panel_merged.csv) — Statistics Denmark panel
- Output: `data/cleaned/housing_agg_district_year.csv`, `data/cleaned/model_panel_full.csv`, `data/cleaned/model_panel_restricted.csv`

---

## Task 0: Notebook structural cleanup (markdown cells only, no code changes)

Insert markdown cells into `EDA_copy.ipynb` at the following positions. No existing code cells are deleted or modified.

**Insert at position 0 (new first cell):**
```markdown
# Copenhagen Housing Price Model
**DTU Model-Based Machine Learning — Spring 2026**

Hierarchical Bayesian model of residential sqm prices across 9 Copenhagen districts (1992–2024),
with a 2026–2060 forecast using Statistics Denmark population projections.

**Data sources**
- Housing transactions: Kaggle Danish Residential Housing Prices 1992–2024
- Socioeconomic panel: Statistics Denmark (KKBEF2, KKLEDIG6, KKIND3, KKHUS1, KKFR2026)
```

**Insert before the housing data load cell (cell id e803dbae):**
```markdown
## Phase 1 · Data Ingestion & EDA
### 1.1 Housing transaction data
Load the Kaggle parquet, filter to Copenhagen zip codes (1000–2720), and map zip codes to the
10 official Statistics Denmark districts.
```

**Insert before the Statistics Denmark load cell (cell id 5b10c8c5):**
```markdown
### 1.2 Statistics Denmark socioeconomic panel
Load five Excel cross-tables from Statistics Denmark (Statistikbanken), reshape from wide
cross-tab format to tidy long format, run quality checks, then pivot back to a district × year
wide panel. Quarterly KKHUS1 household data is averaged to annual frequency.
```

**Insert before the panel merge/save cell (cell id fab807da):**
```markdown
### 1.3 Build historical panel
Pivot each tidy dataset to wide format (district × year) and merge into a single historical panel
(1986–2025). Keep the future population projections (KKFR2026, 2026–2060) as a separate panel.
```

**Insert after the panel merge cell (at the end of Phase 1 content):**
```markdown
## Phase 2 · Feature Engineering
### 2.1 Aggregate housing prices to district-year level
Collapse individual transactions to median log sqm price per district per year.
Filter to regular sales only. Compute within-cell std as observation uncertainty for Pyro.
```

Then add code cells for Task 1, with a separator markdown before Task 2:
```markdown
### 2.2 Align district names and collapse Amager sub-districts
```
And before Task 3:
```markdown
### 2.3 Join housing aggregates with the socioeconomic panel
```

---

## Task 1: Aggregate housing prices to district-year level

**Cell to append to notebook.**

```python
import numpy as np
import matplotlib.pyplot as plt

# 1a. Filter
mask = (
    df['neighborhood'].notna() &
    (df['sales_type'] == 'regular_sale') &
    df['sqm_price'].notna() &
    (df['sqm_price'] > 0)
)
df_filtered = df[mask].copy()

# 1b. Year column + log price
df_filtered['year'] = df_filtered['date'].dt.year
df_filtered['log_sqm_price'] = np.log(df_filtered['sqm_price'])

# 1c. Aggregate
housing_agg = (
    df_filtered
    .groupby(['neighborhood', 'year'])
    .agg(
        median_log_sqm_price=('log_sqm_price', 'median'),
        mean_log_sqm_price=('log_sqm_price', 'mean'),
        std_log_sqm_price=('log_sqm_price', 'std'),
        n_transactions=('sqm_price', 'count'),
        median_sqm=('sqm', 'median'),
        median_purchase_price=('purchase_price', 'median'),
        pct_change_median=('%_change_between_offer_and_purchase', 'median'),
    )
    .reset_index()
    .rename(columns={'neighborhood': 'district'})
)

# 1d. Diagnostics
print(housing_agg.shape)
print(housing_agg.groupby('district')['year'].agg(['min', 'max', 'count']))
print(housing_agg[housing_agg['n_transactions'] < 10])

# 1e. Line plot
fig, ax = plt.subplots(figsize=(12, 6))
for dist, grp in housing_agg.groupby('district'):
    ax.plot(grp['year'], grp['median_log_sqm_price'], label=dist)
ax.legend(fontsize=8); ax.set_title('Median log sqm price by district')
plt.tight_layout(); plt.show()

# 1f. Save
housing_agg.to_csv('data/cleaned/housing_agg_district_year.csv', index=False)
```

---

## Task 2: Align district names (housing_agg) and collapse Amager in historical_panel

**Cell to append after Task 1.**

```python
import pandas as pd

# Load panel if not in memory
historical_panel = pd.read_csv('data/cleaned/historical_panel_merged.csv')

# Show both name sets before any changes
print("housing_agg districts:", sorted(housing_agg['district'].unique()))
print("historical_panel districts:", sorted(historical_panel['district'].unique()))

# --- Part A: rename districts in housing_agg ---
name_map = {
    'Vesterbro': 'Vesterbro/Kongens Enghave',
    # 'Amager' stays as 'Amager' — we will rename the panel side instead
}
housing_agg['district'] = housing_agg['district'].replace(name_map)

# Drop Frederiksberg (separate municipality, not in Statistics Denmark tables)
housing_agg = housing_agg[housing_agg['district'] != 'Frederiksberg'].copy()

# --- Part B: collapse Amager Øst + Amager Vest → 'Amager' in the panel ---
# zip 2300 covers both sub-districts, so housing prices can't be split;
# we aggregate by summing count-type columns for the two Amager rows per year.
amager_rows = historical_panel[historical_panel['district'].isin(['Amager Øst', 'Amager Vest'])]
amager_combined = (
    amager_rows
    .drop(columns='district')
    .groupby('year', as_index=False)
    .sum(numeric_only=True)   # sum counts across the two sub-districts
)
amager_combined.insert(0, 'district', 'Amager')

# Replace the two Amager rows with the combined row
historical_panel = pd.concat([
    historical_panel[~historical_panel['district'].isin(['Amager Øst', 'Amager Vest'])],
    amager_combined
], ignore_index=True)

# --- Verify alignment ---
agg_dists   = set(housing_agg['district'].unique())
panel_dists = set(historical_panel['district'].unique())
unmatched = agg_dists.symmetric_difference(panel_dists)
if unmatched:
    print("UNMATCHED DISTRICTS — stop before merge:", unmatched)
else:
    print("All districts align:", sorted(agg_dists))

# Save both aligned versions
housing_agg.to_csv('data/cleaned/housing_agg_district_year.csv', index=False)
historical_panel.to_csv('data/cleaned/historical_panel_merged_aligned.csv', index=False)
```

**Expected outcome:** 9 districts in both DataFrames: Amager, Bispebjerg, Brønshøj-Husum, Indre By, Nørrebro, Valby, Vanløse, Vesterbro/Kongens Enghave, Østerbro.

---

## Task 3: Join panels

**Cell to append after Task 2.**

```python
import seaborn as sns

# Inner join — use the aligned panel from Task 2
model_panel = housing_agg.merge(historical_panel, on=['district', 'year'], how='inner')

# Diagnostics
print("model_panel.shape:", model_panel.shape)
print(model_panel.groupby('district')['year'].agg(['min', 'max', 'count']))
print("Nulls in key columns:")
print(model_panel[['district', 'year', 'median_log_sqm_price', 'n_transactions']].isnull().sum())

# Coverage per prefix
prefixes = ['unemp_', 'income_', 'pop_', 'housing_']
for pfx in prefixes:
    cols = [c for c in model_panel.columns if c.startswith(pfx)]
    if cols:
        first_non_null_year = model_panel.groupby('district')[cols[0]].apply(
            lambda x: model_panel.loc[x.first_valid_index(), 'year'] if x.first_valid_index() is not None else None
        )
        print(f"{pfx}: first valid year per district =", first_non_null_year.values)

# Missingness heatmap
covariate_cols = [c for c in model_panel.columns if any(c.startswith(p) for p in prefixes)]
miss_frac = (
    model_panel.groupby('district')[covariate_cols]
    .apply(lambda g: g.isnull().mean())
)
# Summarise by prefix for readability
miss_by_prefix = {}
for pfx in prefixes:
    cols = [c for c in covariate_cols if c.startswith(pfx)]
    miss_by_prefix[pfx] = miss_frac[cols].mean(axis=1)
miss_summary = pd.DataFrame(miss_by_prefix)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(miss_summary, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
ax.set_title('Fraction of years with missing values, by district and covariate group')
plt.tight_layout(); plt.show()

# Two panel versions
model_panel_full       = model_panel.copy()
model_panel_restricted = model_panel[model_panel['year'] >= 2008].copy()

# Save
model_panel_full.to_csv('data/cleaned/model_panel_full.csv', index=False)
model_panel_restricted.to_csv('data/cleaned/model_panel_restricted.csv', index=False)

print(f"Full panel:       {model_panel_full.shape[0]} rows, years {model_panel_full['year'].min()}–{model_panel_full['year'].max()}")
print(f"Restricted panel: {model_panel_restricted.shape[0]} rows, years 2008–{model_panel_restricted['year'].max()}")
```

---

## Verification checklist

1. `housing_agg.shape` is approximately (num_districts × num_years) — expect ~9 × 33 ≈ 297 rows (1992–2024)
2. Every district has year coverage from its first available year to 2024, no unexpected gaps
3. No district-years with `n_transactions < 10` among the districts kept (or flagged for awareness)
4. Line plot shows smooth upward trends per district with no obvious data errors
5. After alignment: `set(housing_agg['district'].unique())` exactly matches `set(historical_panel['district'].unique())` — both have 9 districts including `Amager` (combined)
6. After inner join: all 9 districts present, `model_panel_restricted.shape[0]` ≈ 9 × 17 = 153 rows
7. `median_log_sqm_price` and `n_transactions` have zero nulls in both panel versions
8. The missingness heatmap shows `unemp_` columns are all-null before 2008, income/pop/housing available earlier
