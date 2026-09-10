# BPR Financial Distress Early Warning: Data Audit and Modeling Plan

Audit date: 2026-09-11

## Executive decision

The repository contains enough annual data to prototype a next-period financial-deterioration model for conventional BPRs, but it is not ready for modeling without a reproducible cleaning and validation layer.

The first model should:

- cover conventional BPRs only;
- predict a clearly named **next-year multi-indicator deterioration proxy**, not default, failure, license revocation, or supervisory distress;
- use expanding-window, out-of-time validation;
- test peer and spatial variables as incremental features after validating the coordinates and testing spatial dependence; and
- keep BPRS as a separate population and schema, initially outside the primary model.

No genuine distress-event label is present in the repository. A later event-label track should join dated license-revocation or liquidation records from an authoritative OJK/LPS source and preserve the event's effective date and source URL.

## 1. Repository and data-flow audit

The repository has no package, README, environment specification, tests, configuration, or scripted pipeline. It consists of six notebooks and retained CSV/XLSX outputs under `Client-BPRSK/BPRSK`.

### Financial extraction notebooks

| Notebook | Intended scope | Observed design |
|---|---|---|
| `bprkonven1018.ipynb` | Conventional BPR, 2010-2018 | Queries December annual reports from OJK CFS, parses balance sheet, income statement, and ratios, then concatenates with later data. |
| `bprkonven1924.ipynb` | Conventional BPR, 2019-2024 | Current parameter cell says 2020-2024; 2019 is loaded from another retained run. Uses different OJK report codes and parsing logic from the earlier notebook. |
| `bprs1018.ipynb` | BPRS, 2010-2018 | Separate Islamic-bank statement schema and report codes. |
| `bprs1924.ipynb` | BPRS, 2019-2024 | Current parameter cell scrapes only 2024 and appends it to an existing final file. Uses a fixed HTML table index. |

All financial notebooks query `https://cfs.ojk.go.id/cfs/ReportViewerForm.aspx` using bank code, bank name, December, year, annual period type `R`, and a report-type code. They parse HTML tables using keyword matching. The conventional post-2019 notebook selects a table heuristically; the BPRS post-2019 notebook assumes table index 16.

The roster used for every historical year is a single retained bank list rather than a historical year-specific population. The scraper creates one row for every roster-bank/year combination even when no report is recovered. Consequently, the rectangular files are **pseudo-balanced skeletons**, not evidence that each bank operated or reported in every year.

The notebooks initialize a row as failed, but mark it successful after the request sequence even when one or all returned tables are absent. They then drop the status column from exports. Source URL, retrieval timestamp, HTTP status, report-level success, parser version, and parse warnings are not retained. This prevents a reliable distinction among true zero, unavailable report, parser failure, and bank inactivity.

### Geographic notebooks

`longlat.ipynb` geocodes bank name plus city/regency plus province through Google Maps/Selenium. `filterlonglat.ipynb` checks whether results fall on land and reports that 57 of 2,055 initial points were in the ocean before the retained file was completed/corrected. There is no source address, match score, place identifier, or manual-verification flag in the final coordinate table.

The final financial merge uses bank name only, not bank code and not the full name-region key. That creates ambiguous matches and, for BPRS, many-to-many row multiplication.

## 2. Available variables and natural grain

### Conventional BPR

Primary retained file: `Client-BPRSK/BPRSK/FinalBPRK.csv`

- 27,945 rows and 33 columns.
- 1,863 unique bank codes.
- 15 annual December periods, 2010-2024.
- Exactly 1,863 skeleton rows per year and no duplicate `(Kode_Bank, Tahun)` keys.
- Actual non-null total assets range from 1,094 to 1,631 banks in 2010-2023 and 1,348 in 2024.
- 584 banks have at least one intermittent gap in observed total assets.
- Location coverage in the panel: 33 provinces and 297 city/regency labels.

Identifiers and geography:

- `Kode_Bank`, `Nama_BPR`
- `Provinsi`, `Kabupaten_Kota`
- static `Latitude`, `Longitude` joined from the geocoded bank table

Balance-sheet and flow variables:

- `Total_Aset`
- loans to BPRs, commercial banks, related non-banks, and unrelated non-banks
- `Cadangan_Kerugian_Penurunan_Nilai`, `Jumlah_Kredit`
- `Tabungan`, `Deposito`, `Total_Liabilitas`
- `Laba_tahun_berjalan`, `Total_Ekuitas`
- `Agunan_yang_Diambil_Alih`
- interest income, operating income, operating expense, and operating profit/loss

Reported ratios:

- `NPL_Gross`, `NPL_Neto`
- `ROA`, `BOPO`, `NIM`, `LDR`, `KPMM`, `Cash_Ratio`

Ratio coverage is not uniform. Conventional BPR data contain no usable reported ratios in 2010-2012. Complete NPL-net/ROA/BOPO/KPMM rows begin in 2013: 971 complete rows in 2013, generally 1,274-1,519 per year thereafter, and 1,347 in 2024. There are 13,981 complete consecutive bank-year transitions for those four ratios from feature years 2013-2023.

### BPRS

Best pre-merge retained file: `Client-BPRSK/BPRSK/Main/Full1024BPRS.csv`

- 2,910 rows and 43 columns.
- 190 unique bank codes represented by a 194-row roster.
- 15 annual December periods, 2010-2024.
- 60 duplicate `(Kode_Bank, Tahun)` rows caused by four duplicated bank codes assigned conflicting city/regency values in the roster.
- The final name-only coordinate merge increases this to 3,030 rows and 180 duplicate bank-year rows; `FinalBPRS.csv` must not be modeled as-is.
- Location coverage: 25 provinces and 109 city/regency labels.

The BPRS schema contains Islamic financing components, Wadiah/Mudharabah deposits, provisions, liabilities, income/expense variables, and `KPMM`, `NPF_Neto`, `NPF_Gross`, `ROA`, `BOPO`, `FDR`, `Cash_Ratio`, and `NI`.

Pre-2019 and post-2019 fields are not semantically aligned. Examples:

- `BOPO` and `NPF_Gross` are entirely absent before 2019/2023 respectively.
- pre-2019 `Cash_Ratio` contains monetary-scale values rather than percentages;
- several columns are entirely null before 2019 because the statement layout did not expose them; and
- only about 143-153 complete four-ratio transitions are available per modern validation year.

BPR and BPRS should not be pooled mechanically. They require separate feature mappings, target names (NPL versus NPF), and potentially separate models.

### Geographic table

`Client-BPRSK/BPRSK/Main/BankLongLat.csv` contains:

- 2,055 rows, 2,051 unique names, 33 provinces, and 311 city/regency labels;
- bank name, province, city/regency, bank type, latitude, and longitude;
- no missing retained coordinates; and
- 241 rows involved in shared exact coordinates across 55 coordinate pairs. The two largest identical-coordinate groups contain 53 and 51 banks.

Exact coordinate reuse at that scale is consistent with fallback/centroid or ambiguous search matches and must be audited before distance-based weights are trusted.

## 3. Proposed target

### Primary target: next-year material multi-indicator deterioration

For conventional BPR bank `i` observed in year `t`, define four adverse one-year movements from `t` to `t+1`:

1. NPL net increases by at least 2 percentage points;
2. ROA decreases by at least 1 percentage point;
3. BOPO increases by at least 5 percentage points; and
4. KPMM decreases by at least 5 percentage points.

Set `next_period_deterioration = 1` when at least two of the four adverse movements occur, requiring all four indicators at both endpoints for the primary complete-case target. Otherwise set it to zero. Preserve the count and individual component flags so the binary result remains auditable.

This is a proposed, pre-model specification, not yet an implemented label. The thresholds are interpretable percentage-point moves and broadly correspond to adverse upper-quartile movements in the observed historical BPR changes. They must be subjected to sensitivity analysis (for example, one/two/three component rules and nearby thresholds) using development years only. They must not be tuned against the final holdout's model performance.

Why this target is suitable:

- it is forward-looking: information through `t` predicts a change realized at `t+1`;
- it spans asset quality, profitability, efficiency, and capital rather than equating one volatile ratio with distress;
- it avoids unusable conventional `NPL_Gross` values, which are systematically zero in 2019-2022; and
- each positive can be explained by its realized component deteriorations.

Limitations:

- it is a relative financial-deterioration proxy, not a regulatory distress or failure label;
- annual December data cannot establish when within the year deterioration began;
- thresholds encode materiality choices and require stability testing;
- measurement/parser changes can create false movements; and
- requiring next-year observations excludes banks that disappear, so selection bias must be measured rather than interpreting disappearance as either good or bad.

Maintain a continuous companion outcome—the count or standardized severity of adverse changes—for sensitivity analysis and ranking. Do not call the model a probability of default. Label outputs as `probability_of_next_year_deterioration`.

### Future genuine-event target

Build a separate dated event registry only when authoritative external records are collected. Required fields are bank code/name, event type, announcement date, effective date, source authority, source URL/document, and reconciliation status. A discrete-time hazard target can then represent entry into a first verified distress event. The event date must be after the observation cutoff, and renamed/merged banks require a point-in-time identity crosswalk.

## 4. Major data-quality issues and required gates

1. **2024 monetary unit break.** Median within-bank ratios of 2024 to 2023 are about 1,049 for total assets, 1,056 for loans, 1,056 for savings, 1,057 for deposits, and 1,044 for equity. BPRS shows the same approximately 1,000-fold shift. Monetary values must be reconciled to one documented unit before 2024 trend or growth features are built.
2. **Missingness is not zero.** `clean_number` returns zero for missing or unparsable values. Some ratio zeros are later changed to blanks, while balance-sheet zeros are retained. Re-extraction must preserve raw text and separate `reported_zero`, `not_reported`, `not_applicable`, and `parse_error`.
3. **False success status.** A bank-year can be marked successful even when all report tables are missing, and status is removed from final CSVs.
4. **Current-roster backfill.** A single roster is crossed with all historical years. This creates survivorship/population ambiguity and potentially informative missingness.
5. **Schema/parser breaks.** Report codes and parsers change around 2013 and 2019; 2024 has a unit change. These boundaries require explicit schema versions and stability checks.
6. **Implausible ratio tails.** Conventional maxima include NPL net 1,560%, ROA 989%, BOPO 8,580%, LDR 7,550%, KPMM 7,250%, and cash ratio 10,000%. These values must be checked against raw reports, not blindly winsorized.
7. **NPL field anomaly.** Conventional `NPL_Gross` is zero for the majority of 2019-2022 observations while `NPL_Neto` is populated. This is consistent with a parser miss, not genuine zero risk.
8. **BPRS duplicate identities.** Four codes occur twice with conflicting cities; the subsequent name-only location join multiplies rows.
9. **Geocode uncertainty.** Coordinates are search-derived without match confidence or source address; large exact-coordinate clusters make fine-distance peer definitions unreliable until verified.
10. **Non-reproducible paths and outputs.** Notebooks contain absolute local paths, timestamped intermediate filenames not retained in the repository, manual concatenations, and no data dictionary or checksums.
11. **Units are undocumented.** The monetary unit and any OJK presentation-unit changes need to be read from the source report metadata and stored by report/version.
12. **No external regional data or event labels.** Province/city labels exist, but BPS/OJK regional economics and genuine adverse-event data do not.

Modeling is a no-go until issues 1-4 and the bank/location key problem are repaired. Ratio-only exploratory target profiling can proceed after validating a sample of source reports at each schema boundary.

## 5. Temporal validation strategy

Each modeling row is indexed by bank and observation year `t`; its outcome is realized in `t+1`. Splits are defined by **target year**, not random rows.

Recommended expanding-window evaluation for the conventional BPR proxy:

| Fold | Train target years | Validation target year | Purpose |
|---|---|---|---|
| 1 | 2016-2018 | 2019 | Explicit report-schema boundary stress test |
| 2 | 2016-2019 | 2020 | COVID-era shift |
| 3 | 2016-2020 | 2021 | Out-of-time test |
| 4 | 2016-2021 | 2022 | Out-of-time test |
| 5 | 2016-2022 | 2023 | Recent out-of-time test |
| Final | 2016-2023 | 2024 | Locked final holdout |

The 2016 start permits three annual observations for slope/volatility features. Current-condition models can use earlier rows, but ablations must compare identical evaluation cohorts. Report both the schema-boundary fold and a modern-era summary; do not hide a weak boundary result inside an average.

All preprocessing—imputation, clipping bounds, scaling, target sensitivity decisions, calibration, feature selection, and risk-band thresholds—is fitted on each fold's training window only. Peer features for row `(i,t)` use peers' values no later than `t`. No statistic calculated from `t+1` can enter features.

Use the latest 2024 observations only for operational scoring toward 2025 after the monetary-unit repair. Such scores have no realized target yet and must not appear in performance evaluation.

## 6. Feature families

Feature definitions should be versioned and few enough to explain financially.

### Bank-specific current condition

- asset quality: NPL net initially; NPL gross only after parser repair;
- profitability: ROA, operating profit/assets, current-year profit/assets;
- efficiency: BOPO and operating expense/operating income;
- capital: KPMM, equity/assets, and a documented capital-buffer feature only after the applicable regulatory threshold is sourced by year;
- liquidity/funding: LDR, loans/(savings + deposits), deposit mix, cash ratio;
- scale: log assets, log loans, deposits, equity, and provisions; and
- coverage/collateral: provisions/loans and acquired collateral/assets.

### Temporal deterioration

- one-year percentage-point changes for reported ratios;
- log growth for positive monetary stocks after unit harmonization;
- two- and three-year rolling means and volatility where history is truly observed;
- three-year least-squares slope for NPL net, ROA, BOPO, KPMM, and funding growth;
- second difference only for core ratios with three valid consecutive observations;
- consecutive adverse-movement counts; and
- deviation from the bank's expanding historical median, calculated using years through `t` only.

### Interpretable balance-sheet relationships

- loan growth minus deposit growth;
- NPL-net change alongside loan growth, represented as an interaction rather than an invented stock ratio;
- provision coverage = provisions/loans;
- profitability versus asset growth interaction;
- equity/assets and, when sourced, KPMM buffer above the applicable minimum;
- funding concentration = deposits/(savings + deposits); and
- liquidity pressure combining high LDR with negative deposit growth.

Missingness and parser-quality flags should be retained for monitoring, but failure-to-report features must not be used predictively until their operational meaning and availability at scoring time are established.

## 7. Peer and spatial methodology

### Identity and coordinate gate

Create one canonical bank table keyed by bank code and bank type. Resolve the four BPRS conflicts. Match coordinates using code where possible or a reviewed name-region crosswalk, never name alone. Add coordinate source, confidence, precision class (rooftop, street, city centroid, unknown), validation flag, and as-of date.

### Exploratory diagnostics

For each sufficiently populated year, test Global Moran's I on NPL/NPF net, ROA, BOPO, and the continuous deterioration score. Use permutation inference and report effect sizes and multiple-testing-adjusted results. Use Local Moran/LISA for diagnosis and maps, not as a future-informed model label. Repeat tests under more than one defensible weights definition.

### Candidate weights

1. Same city/regency leave-one-bank-out peers—the most reliable initial definition given coordinate uncertainty.
2. Same province leave-one-bank-out peers as a broader robustness check.
3. Coordinate k-nearest neighbors after geocode validation, with a small fixed `k` chosen in training only and great-circle distance.
4. Distance-decay weights after validating coordinate precision, with self-weight zero and row normalization.

Administrative neighbors require an external boundary/crosswalk dataset and should not be inferred from labels alone.

### Predictive peer/spatial features

At observation year `t`, compute leave-one-bank-out peer median/mean NPL net, ROA, and BOPO; peer one-year changes; share of peers whose latest observed movement deteriorated; own-minus-peer median; within-region percentile; neighbor count; local asset concentration; and distance-weighted peer risk. Require minimum peer counts and expose missing/low-support flags.

No peer feature at `t` may use a peer's `t+1` outcome. If reports have differing publication dates, an operational version should use actual availability timestamps rather than assuming all December reports were known simultaneously.

Spatial information earns a place only if Moran/LISA diagnostics are credible and out-of-time ablation improves discrimination, calibration, or top-decile capture. Otherwise retain administrative peer benchmarking and report that fine geographic proximity did not add value.

## 8. Regional enrichment interface

No regional macroeconomic data are currently present. Add a future point-in-time table keyed by normalized BPS region code and period, with separate `available_at` and `reference_period` fields. Candidate sourced variables include GRDP growth, unemployment, poverty, inflation, sector shares, population, regional credit growth, and institution density.

The interface must include source agency, dataset/table identifier, source URL, geographic vintage, unit, release date, transformation, and revision vintage. Joining by free-text province/city is not acceptable; build a versioned crosswalk to BPS administrative codes. Lag data according to publication availability at the prediction cutoff.

Conceptually keep feature namespaces separate:

- `bank__*`: own financial condition and trends;
- `region__*`: regional economic context; and
- `peer__*` / `spatial__*`: contemporaneous leave-one-out peer conditions.

## 9. Statistical and ML benchmark plan

Use identical folds and eligible rows for every ablation.

1. Intercept/year-only benchmark and simple last-condition score.
2. Regularized logistic regression with standardized numeric inputs and transparent coefficients.
3. Interpretable nonlinear benchmark using splines/GAM if sample size and dependency support are adequate.
4. Discrete-time hazard model only for a future verified first-event target; the deterioration proxy is recurrent and should not be mislabeled as first-event survival.
5. Random-intercept or correlated-data sensitivity model for repeated banks, provided estimation is stable. Bank fixed effects are unsuitable for scoring unseen or short-history banks and absorb much of the signal.
6. LightGBM or CatBoost as the nonlinear benchmark, not the default champion.

Ablations:

1. current financial variables;
2. current financial plus temporal trends;
3. previous layer plus administrative peers;
4. previous layer plus validated coordinate-spatial features; and
5. previous layer plus point-in-time regional variables when sourced.

Evaluate ROC-AUC, PR-AUC, KS, recall, precision, Brier score, calibration intercept/slope and plots, event rate and lift by decile, and top-10%/top-20% capture. Report annual metrics and their dispersion, not only a pooled score. Bootstrap uncertainty by bank rather than individual row.

Risk bands should be selected from training-fold operational capacity and observed deterioration rates—for example, review-capacity cutoffs for the top risk population—then frozen for the next validation year. Report migration matrices and realized rates by band. Do not assign arbitrary probability cutoffs such as 0.25/0.50/0.75.

Global explanations should include coefficient/spline plots and permutation or SHAP summaries. Bank-level explanations must be generated from the fitted model and decompose own-condition, trend, peer/spatial, and regional contributions. Explanations must state data-quality limitations and never translate proxy deterioration into default probability.

## 10. Leakage and invalid-assumption register

| Risk | Failure mode | Control |
|---|---|---|
| Random cross-validation | Same economic regimes and repeated banks leak across folds | Split strictly by target year; cluster uncertainty by bank |
| Forward target leakage | `t+1` ratios or deterioration components enter features | Central cutoff-aware feature builder and tests |
| Peer leakage | Peer outcomes from `t+1` are averaged into row `t` | Group peer features by observation year only and shift peer trends correctly |
| Global preprocessing | Future medians, clipping, scaling, or calibration influence training | Fit every transform inside each training fold |
| Historical baseline leakage | Full-history bank average includes future years | Expanding/rolling calculations ending at `t` |
| Geographic leakage | LISA cluster based on future outcome becomes a feature | Diagnostics are year-specific; predictive lags use `t` only |
| Publication timing | December peer/macroeconomic values were not available at scoring cutoff | Add `available_at` and simulate a realistic cutoff |
| Attrition-as-default | Missing next report is treated as distress without an event source | Keep outcome unknown; analyze selection separately |
| Current-roster bias | Historical population is defined using a later roster | Recover year-specific rosters or document cohort conditioning |
| Duplicate merge | Name-only join multiplies BPRS observations | Canonical code-based identity and one-to-one merge assertions |
| Unit/schema shift | 2024 growth or historical trends become artificial | Versioned unit normalization and boundary regression tests |
| Target tuning | Proxy thresholds selected for best holdout AUC | Pre-register thresholds; sensitivity analysis on development years only |
| Spatial overclaim | Geocoder errors create artificial neighbor structure | Precision audit, alternative weights, permutation tests, ablation |

## 11. Implementation sequence after approval

1. Preserve current notebooks and raw retained files as legacy inputs.
2. Add a data dictionary, provenance manifest, schema versions, and raw/clean validation reports.
3. Implement canonical BPR/BPRS identities and safe coordinate joins with one-to-one assertions.
4. Reconcile monetary units and validate sampled raw OJK reports at 2012/2013, 2018/2019, and 2023/2024 boundaries.
5. Build a conventional-BPR clean panel and target audit; publish prevalence and stability before fitting models.
6. Implement cutoff-safe financial and temporal features with unit tests.
7. Run spatial diagnostics; build peer/spatial features only if supported.
8. Implement expanding-window baselines, calibration, ablations, and risk-oriented evaluation.
9. Produce the bank risk table and monitoring artifacts from the latest eligible observation period.
10. Add external event and regional enrichment only through versioned, point-in-time interfaces.

The immediate next milestone is therefore **data repair plus target validation**, not model training.
