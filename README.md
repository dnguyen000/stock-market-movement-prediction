# SEIS-763: Midterm Project — Stock Market Movement Prediction

**Team Members:** Brahmee Adhikari, Don C. Nguyen

---

## Goal

This project builds machine learning models to predict stock price movement using historical daily price data. Two tasks are performed:

1. **Classification** — Predict whether a stock's price will go **UP or DOWN** the next trading day.
2. **Regression** — Predict **how much the price will change** over the next 5 trading days.

Both tasks are applied to two tickers:
- **SPY** — An S&P 500 ETF representing a stable, broad-market benchmark
- **TSLA** — Tesla, Inc., a high-volatility individual stock

Comparing the two tickers reveals whether stability or volatility makes a stock easier to predict, and how extreme market events affect model performance.

---

### Tools

- Claude AI: used to:
  - Help understand stock concepts and terminology
  - Create a feedback loop on design ideas, clarify logic in code, support/criticize optimization ideas, and validate observations made during the project
  - Automate documentation of results and clean summarizations of user prompted notes taken of model results
- Class notes: used as reference to validate concepts to be demonstrated in this project

---

## Data Source

Historical daily stock price data is sourced from the **Yahoo Finance API (yfinance)**. Data spans **10 years** of trading days (2015–2024) for both SPY and TSLA (~2,515 rows each). Each trading day includes the following native attributes:

| Attribute | Description |
|---|---|
| `open` | Price at market open |
| `high` | Highest price of the day |
| `low` | Lowest price of the day |
| `close` | Unadjusted price at market close |
| `adj_close` | Dividend-adjusted closing price (relevant for SPY) |
| `volume` | Number of shares traded |

From these attributes, additional features are derived (see Key Terms below).

---

## Key Terms

### Stock Market Terms

| Term | Definition |
|---|---|
| **Daily Return** | The percentage change in a stock's closing price from one day to the next. Captures short-term momentum. |
| **Weekly Return** | The percentage change in closing price over the past 5 trading days (~1 week). |
| **Moving Average (20-day)** | The average closing price over the past 20 trading days. Smooths out noise and reveals the medium-term trend. |
| **Distance from 20-day MA** | How far today's price is above or below the 20-day moving average, expressed as a percentage. Indicates whether a stock is overbought or oversold relative to its recent trend. |
| **Daily Range** | The difference between a day's high and low price, normalized by the closing price. Measures intraday volatility. |
| **Volume** | The total number of shares traded in a day. High volume often signals strong conviction behind a price move. |
| **Volume Change** | The day-over-day percentage change in trading volume. |
| **VWAP (Volume Weighted Average Price)** | The average price a stock traded at throughout the day, weighted by volume. A common institutional benchmark. |
| **VWAP Distance** | How far the closing price is from VWAP. Indicates whether the stock closed above or below the average trade price. |
| **Volatility (20-day)** | The rolling standard deviation of daily returns over 20 days. Higher values mean more unpredictable price swings. |
| **RSI (Relative Strength Index)** | A momentum indicator scaled 0–100. Values above 70 suggest a stock is overbought; below 30 suggests oversold. Calculated over 14 days. |
| **Bollinger Band Position** | Where today's closing price sits within the Bollinger Bands, which are set 2 standard deviations above and below the 20-day moving average. Values near +1 or -1 suggest price is at the edge of its recent range. |
| **VIX (CBOE Volatility Index)** | A real-time index published by the Chicago Board Options Exchange (CBOE) that measures the market's expectation of volatility over the next 30 days. It is derived from the implied volatility of S&P 500 options — when traders are willing to pay more for options (i.e., hedging against large swings), VIX rises. Often called the "fear gauge," VIX tends to spike sharply during crises and remain elevated until conditions stabilize. A VIX above 30 is a widely used threshold for heightened market stress. In this project, VIX daily closing prices are fetched from Yahoo Finance (`^VIX`) and merged into the feature dataset to derive the extreme event indicator. |
| **Extreme Event Indicator** | A boolean (True/False) flag marking trading days where the VIX closing price exceeded 30, indicating a period of elevated market fear or stress. Rather than hardcoding specific crash dates, this approach is data-driven: it automatically captures any shock period in the dataset where volatility was abnormally high. Helps the model distinguish genuine market disruptions from normal day-to-day noise. |
| **SPY** | The SPDR S&P 500 ETF Trust. Tracks the S&P 500 index, representing 500 large U.S. companies. Used here as a stable market benchmark. |
| **TSLA** | Tesla, Inc. A high-growth, high-volatility individual stock used here as a contrast to SPY. |

**Note**:
When considering the data sources, we made the decision to keep the dataset within what is provided through the API and creating derived values from those standard values. This was to ensure we stayed within the timeline of the project and could deliver a finished product.

---

### Process
Each fold trains the model exclusively on its training window, then predicts the immediately following test window. An embargo separates the two to prevent feature leakage from rolling window calculations. The next fold begins immediately after the previous test window ends:

This approach captures performance across multiple market conditions rather than a single arbitrary split.

### Metrics Reported
Results are aggregated as **mean ± standard deviation** across all folds.

| Task | Primary Metric | Supporting Metrics |
|---|---|---|
| Regression | RMSE | MAE, R² |
| Classification | F1 Score | Accuracy, Precision, Recall, Specificity, Confusion Matrix |

Standard deviation across folds measures **consistency** — a model with lower std is more reliable across different market conditions.

---

## Training / Testing

### Classification

All classification models use a shared feature set of **19 engineered features** derived from price, volume, momentum, volatility, and market-stress signals. Each model is evaluated with walk-forward validation using `class_weight='balanced'` to mitigate the natural imbalance between UP and DOWN days. Features are standardized via `StandardScaler` fit inside each training fold only (the scaler is never fit on test data). All models use `random_state=42` for reproducibility. SVM additionally enables `probability=True` for probability calibration.

Standard configuration: train=189d / test=42d / embargo=5d (SPY: 55 folds; TSLA: train=59d / 58 folds, except Random Forest which uses 189d/55 folds for both tickers).

#### Hyperparameter Selection Workflow (Three-Way Comparison)

For every base model on every ticker, we ran a three-way comparison rather than picking the grid search winner outright:

1. **Baseline** — an initial reasonable hyperparameter choice, used as a reference point before any tuning.
2. **Best F1 (grid search winner)** — the configuration with the highest mean F1 across walk-forward folds.
3. **Balanced** — a separately chosen configuration that traded some F1 in exchange for meaningfully higher specificity, addressing the class-bias issue described in the Key Findings below.

This workflow was adopted after the first grid search (SPY SVM) revealed that the best-F1 configurations consistently achieved their scores by biasing predictions toward the UP class. For example, SPY SVM's best-F1 grid point (C=1, γ=1) reached F1 = 0.608 with specificity of only 0.189 — catching 82% of UP days but missing 81% of DOWN days. The balanced configuration sacrifices a few points of F1 to keep specificity in the 0.45–0.50 range, producing a more honest classifier. Voting ensemble components were chosen from the balanced configurations, except where the F1 vs. balanced trade-off was negligible (e.g., TSLA Logistic Regression, where we kept the slightly higher-F1 L1 variant).

#### Models

Predict the **next-day price direction** — whether the stock's closing price will be higher (UP) or lower (DOWN) than the prior day's close.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Logistic Regression** | Linear baseline classifier. Fits a logistic function to the feature space to output a probability of UP. L1 and L2 regularization are compared via grid search. Provides interpretable coefficients but is limited to linear decision boundaries. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Random Forest** | Bagging ensemble of decision trees. Each tree is trained on a bootstrap sample of the training window; majority vote across trees determines the prediction. Captures non-linear interactions between features without requiring explicit kernel specification. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Support Vector Machine (SVM)** | Kernel-based classifier using an RBF (radial basis function) kernel. Finds the maximum-margin hyperplane in a high-dimensional feature space, allowing it to capture non-linear boundaries. Regularization strength (C) and kernel width (gamma) are tuned via grid search. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **XGBoost** | Gradient boosting ensemble that builds decision trees sequentially, each correcting the residual errors of the prior tree. Uses `scale_pos_weight` to handle class imbalance between UP and DOWN days. Provides feature importance scores ranking which technical indicators most influence directional predictions. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Voting Ensemble (Hard Vote)** | Combines all four classifiers above using a hard majority vote. Components are drawn from the balanced configurations identified by the three-way comparison workflow. Two tie-breaking thresholds are evaluated: ≥2 of 4 (tie breaks toward UP) and ≥3 of 4 (tie breaks toward DOWN). | F1 Score | Accuracy, Precision, Recall, Specificity |

**Exploratory variants.** Four notebooks (`rf-Copy1_with_17_features.ipynb`, `rf-tsla-Copy1_with_17_features.ipynb`, `xboost_spy-Copy1_witth_17_features.ipynb`, `xboost_tsla-Copy1_with_17_features.ipynb`) test reduced 16-feature variants (dropping `month`, `is_earnings_week`, `is_major_event`) to assess whether trimming low-importance features improves performance. Results were within fold-level noise of the canonical 19-feature models, so the trimmed variants were not used in final reporting. These notebooks are retained for reference in `classification/exploratory/` but are not part of the production pipeline.

### Regression

All models are evaluated using walk-forward validation across both tickers independently.

#### Window Configurations Across Iterations

| Iteration | Train | Test | Embargo | Approx. Folds | Dataset |
|---|---|---|---|---|---|
| 1 | 63d | 42d | 5d | ~58 | Yahoo Finance, 10yr |
| **2** | **189d (SPY) / 59d (TSLA)** | **42d** | **5d** | **55 (SPY) / 58 (TSLA)** | **Yahoo Finance, 10yr** |
| 3 | 3d | 1d | 0d | ~2,507 | Yahoo Finance, 10yr |
| 4 | 15d | 5d | 2d | ~498 | Yahoo Finance, 10yr |

Iteration 2 is the primary production configuration. Iterations 3 and 4 explore the effect of shorter windows and are included as cautionary comparisons.

> **Note on scope:** All classification experiments were conducted under Iteration 2 only (SPY 189d / TSLA 59d, 42d test, 5d embargo). Iterations 1, 3, and 4 apply to the regression experiments — see the Regression section.

#### Models
Predict the **5-day forward return** — the percentage price change from today's close to the close 5 trading days from now. A positive value means the model expects the price to rise; negative means it expects a decline.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Linear Regression** | Baseline model. Fits a straight line through the feature space to predict the 5-day return as a weighted sum of inputs. Coefficients reveal which features most influence the prediction. | RMSE | MAE, R² |
| **Ridge Regression** | Extension of linear regression with L2 regularization — penalizes large coefficients to reduce overfitting. Particularly useful here because technical indicators (RSI, Bollinger Band position, distance from MA) are often correlated with each other. | RMSE | MAE, R² |
| **XGBoost** | Gradient boosting ensemble that builds decision trees sequentially, each correcting the residual errors of the prior tree. Captures non-linear relationships between features and returns that linear models cannot, and provides feature importance scores. | RMSE | MAE, R² |

---

## Results

### Classification

Results are reported as **mean ± standard deviation** across all walk-forward folds. For granular per-model tables, grid search results, and seasonal breakdowns, see:
- [SPY Classification Results](results/spy_classification_results.md)
- [TSLA Classification Results](results/tsla_classification_results.md)

Walk-forward config: train=189d / test=42d / embargo=5d (SPY: 55 folds; TSLA standard: train=59d / 58 folds)

#### Best Model — SPY: Voting Ensemble (Hard Vote)

Ensemble components were chosen via the three-way comparison workflow described in the Models section above. For SPY, all four components are drawn from the **balanced** configurations:

| Model | Selected Config | Source |
|---|---|---|
| SVM | C=100, γ=0.1 | Balanced |
| Logistic Regression | C=0.001, L2 | Balanced |
| Random Forest | n_est=200, depth=6, min_leaf=5, sqrt | Balanced |
| XGBoost | n_est=500, depth=4, lr=0.01, subsample=0.8 | Balanced |

##### Tie-breaking sensitivity

The ensemble uses a hard majority vote across 4 models. With 4 models, ties (2 UP / 2 DOWN) are possible. We evaluated two tie-breaking rules:

- **≥ 2 of 4 (tie → UP):** predict UP whenever any two models agree
- **≥ 3 of 4 (tie → DOWN):** predict UP only when three or more models agree

| Metric | Tie → UP (≥ 2 of 4) | Tie → DOWN (≥ 3 of 4) |
|---|---|---|
| Accuracy | 0.521 ± 0.091 | 0.504 ± 0.079 |
| F1 (UP) | 0.574 ± 0.119 | 0.478 ± 0.152 |
| Precision (UP) | 0.567 | 0.568 |
| Recall (UP) | 0.628 | 0.463 |
| Specificity (DOWN) | 0.397 | 0.560 |
| Macro F1 | 0.51 | 0.50 |

**Interpretation.** The "tie → UP" rule produces a higher F1 by predicting UP more often — catching more actual UP days (recall 0.628) at the cost of missing DOWN days (specificity 0.397). The "tie → DOWN" rule is the mirror image: predicts UP less often, catches more DOWN days (specificity 0.560), at the cost of UP recall (0.463). Precision is essentially identical (0.567 vs 0.568) — when either rule says UP, it is correct equally often. Macro F1 is functionally identical between the two rules (~0.50), confirming that the two configurations sit at the same overall predictive ceiling and differ only in **which class absorbs the errors**.

This independently reproduces the class-bias pattern seen at the per-model grid search level: the same underlying model can produce F1 of 0.574 or 0.478 depending on a single tie-breaking choice, without any change in actual predictive ability. We report the "tie → UP" rule as the primary best-F1 configuration for consistency with the headline metric, while noting the "tie → DOWN" rule reflects the model's balanced classification capability more honestly.

**Seasonal breakdown (tie → UP):**

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 575 | 0.577 |
| Summer | 580 | 0.605 |
| Fall | 603 | 0.587 |
| Winter | 552 | 0.591 |

#### Best Model — TSLA: Voting Ensemble (Hard Vote)

Ensemble components follow the same three-way comparison workflow. For TSLA, three components are drawn from the balanced configurations; Logistic Regression uses the Best F1 variant because the F1 vs. balanced trade-off was negligible at that model:

| Model | Selected Config | Source |
|---|---|---|
| SVM | C=10, γ=0.1 | Baseline (no grid F1 improvement over baseline) |
| Logistic Regression | C=10, L1 | Best F1 |
| Random Forest | n_est=500, depth=4, min_leaf=10, sqrt | Balanced |
| XGBoost | n_est=100, depth=3, lr=0.01, subsample=1.0 | Balanced |

##### Tie-breaking sensitivity

As with SPY, we evaluated two tie-breaking rules for the 4-model majority vote:

| Metric | Tie → UP (≥ 2 of 4) | Tie → DOWN (≥ 3 of 4) |
|---|---|---|
| Accuracy | 0.503 ± 0.083 | 0.491 ± 0.074 |
| F1 (UP) | 0.541 ± 0.163 | 0.421 ± 0.191 |
| Precision (UP) | 0.536 | 0.510 |
| Recall (UP) | 0.652 | 0.434 |
| Specificity (DOWN) | 0.365 | 0.579 |
| Macro F1 | 0.49 | 0.49 |

**Interpretation.** The same mirror-image pattern seen on SPY appears more pronounced on TSLA: switching the tie-breaking rule swings F1 by 0.12 (0.541 → 0.421) and inverts recall and specificity, while macro F1 stays at 0.49 in both cases. The larger F1 swing on TSLA (0.12 vs SPY's 0.10) reflects the weaker underlying signal — when actual classification ability is closer to chance, F1 moves more with the decision threshold because there is less real predictive capability to anchor it.

**Seasonal breakdown (tie → UP):**

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 613 | 0.538 |
| Summer | 645 | 0.543 |
| Fall | 630 | 0.610 |
| Winter | 548 | 0.596 |

> **Note on class balance.** TSLA shows the same asymmetric pattern as SPY but more pronounced: under the ≥ 2 rule, UP recall is 0.652 and DOWN recall (specificity) is 0.365. Per-fold analysis (visible in `svm_tsla.ipynb`) shows the SVM frequently collapses to predicting a single class for an entire 42-day test window — alternating between "predict UP for all 42 days" and "predict DOWN for all 42 days" depending on the training slice. Probability distributions for actual UP vs. actual DOWN days overlap almost entirely (mean P(UP) = 0.523 on UP days, 0.519 on DOWN days), indicating the model cannot reliably distinguish the two classes from the available technical features. The reported F1 reflects this bias and should not be read as evidence of balanced classification ability.

### Key Findings Across Classification Models

1. **The F1 trap was visible from the first grid search and shaped every subsequent decision.** Across all four base models on both tickers, the highest-F1 configurations from grid search consistently achieved their scores by biasing predictions toward the UP class. The most extreme example: SPY SVM's best-F1 config (C=1, γ=1) hit F1 = 0.608 but with specificity of only 0.189 — meaning the model caught 82% of UP days but only 19% of DOWN days. To handle this, we adopted a **three-way comparison** workflow for every model: (a) an initial baseline hyperparameter choice, (b) the grid search winner by F1, and (c) a separately chosen "balanced" configuration that traded some F1 for meaningfully higher specificity. Component models for the voting ensemble were then selected from these balanced configurations (or, where the F1 vs balanced trade-off was small, the better F1 option), not from the raw F1 winners.

2. **The voting ensemble's tie-breaking rule independently confirms the F1 trap.** On SPY, switching the voting threshold from ≥ 2 of 4 (tie → UP) to ≥ 3 of 4 (tie → DOWN) changes F1 from 0.574 to 0.478, recall from 0.628 to 0.463, and specificity from 0.397 to 0.560 — while macro F1 (~0.50) and precision (0.567) remain essentially unchanged. The same ensemble produces dramatically different F1 numbers depending on a single threshold, with no underlying change in classification ability. This makes the F1 trap concretely demonstrable end-to-end: it shows up at the per-model level (grid search), at the ensemble level (voting threshold), and both at the same magnitude.

3. **SPY is more predictable than TSLA but only marginally.** All four models landed in the 48–53% accuracy range for both tickers, with F1 between 0.46 and 0.57. SPY's voting ensemble F1 = 0.574 vs TSLA's 0.545 — a real but small gap, well within the published ceiling for daily direction prediction on liquid equities (52–55%). Tree-based models (RF, XGBoost) slightly outperformed linear models (LR, SVM) on average, but no single model dominated across both tickers.

4. **Per-fold variance dominates the picture.** F1 standard deviations of ±0.12 (SPY) and ±0.15 (TSLA) across folds are large relative to the mean F1 itself. Individual folds range from near-zero F1 to F1 above 0.75. This reflects market regime shifts more than model instability — the same model produces very different results in trending vs. choppy market periods.

5. **Seasonal performance is consistent.** F1 across Spring/Summer/Fall/Winter is within 0.03 for SPY and 0.07 for TSLA, indicating no strong seasonal effect dominates the model's behavior. Differences are within fold-level noise.

6. **Methodology limitation — hyperparameter selection bias.** Grid search for each base model was conducted across the same walk-forward folds later used for final evaluation. Both the F1-winner and balanced configurations were selected based on their mean metrics across those folds. This is a form of selection bias: the reported best-model metrics are optimistic relative to what an unbiased held-out test would show. A nested cross-validation scheme (selecting hyperparameters on inner folds, evaluating on outer folds) would address this but was not implemented due to scope constraints. The magnitude of this bias is likely modest — most grid configurations produced F1 within 0.02–0.05 of each other on a given ticker, suggesting model performance is not highly sensitive to specific hyperparameter choices in this region of the search space.

---

### Regression

Results are reported as **mean ± standard deviation** across all walk-forward folds. For granular per-iteration tables and seasonal breakdowns, see:
- [SPY Regression Results](results/spy_regression_results.md)
- [TSLA Regression Results](results/tsla_regression_results.md)

#### SPY (best model per iteration)

| Iteration | Train Window | Best Model | RMSE (mean ± std) | Seasonal R² highlights |
|---|---|---|---|---|
| 1 (5-day return target) | 63d | XGBoost | 0.0262 ± 0.0154 | n/a (different target) |
| **2** | **189d** | **Ridge** | **0.005168 ± 0.004213** | Spring **+0.273**, Fall +0.144, Winter +0.040 |
| 3 | 3d | XGBoost | 0.002181 ± 0.002797 | All seasons positive (interpolation artifact) |
| 4 | 15d | XGBoost | 0.004424 ± 0.004333 | Spring **+0.421**, Fall **+0.178** |

#### TSLA (best model per iteration)

| Iteration | Train Window | Best Model | RMSE (mean ± std) | Seasonal R² highlights |
|---|---|---|---|---|
| 1 (5-day return target) | 63d | XGBoost | 0.1001 ± 0.0395 | n/a (different target) |
| 2 | 59d | XGBoost | 0.019849 ± 0.009929 | All negative; best Summer −0.347 |
| 3 | 3d | XGBoost | 0.008691 ± 0.011084 | All positive (interpolation artifact) |
| 4 | 15d | XGBoost | 0.016236 ± 0.012345 | Only Spring positive: **+0.228** |

### Key Findings Across All Iterations

1. **Iteration 2 (SPY: 189d window) is the strongest well-founded configuration.** Ridge Regression achieves positive R² in three of four seasons for SPY (Spring +0.273, Fall +0.144, Winter +0.040) — the only non-Iter3 configuration with multi-season positive R².

2. **Ridge Regression outperforms XGBoost for SPY at the 189-day window.** Sufficient history allows L2 regularization to converge to stable coefficients. At shorter windows, XGBoost's non-linearity is a bigger advantage.

3. **Fall is consistently learnable for SPY across all informative iterations** (Iter2 Ridge R²=+0.144, Iter4 XGBoost R²=+0.178), confirming it as a genuine signal period rather than a window artifact.

4. **TSLA regression shows no positive overall R² in Iterations 1, 2, or 4.** Regime shifts (meme-stock 2020–2022, post-election 2024) require longer windows than tested.

5. **Iteration 3 (3d/1d) numbers are misleading.** Models interpolate within a nearly static local window rather than generalizing. The high R² values reflect local fitting, not predictive power.

6. **Linear regression collapses at small training windows.** At 15 samples / 16 features (Iter4), R² averages −3,582 for SPY and −257,532 for TSLA.

---

## Conclusion

### Classification

Daily direction classification on both SPY and TSLA performed near the published ceiling for this problem (~52% accuracy for liquid equities, near chance for individual volatile stocks). The Voting Ensemble outperformed all single models on F1 — but as documented in the per-ticker class-balance notes, F1 gains are partly attributable to the model biasing toward the UP majority class. Macro F1 paints a more conservative picture and is closer to balanced-accuracy levels around 0.49–0.50 for both tickers.

The central methodological finding from the classification track is the **F1 trap** and the workflow we developed to handle it. Grid search across every base model consistently surfaced configurations with inflated F1 driven by class bias rather than genuine classification ability. The three-way comparison workflow (baseline / best-F1 / balanced) made this trade-off explicit at every step, and ensemble components were chosen from the balanced side of that trade-off rather than the raw F1 winners. The voting threshold sensitivity analysis (≥2 vs ≥3 of 4) independently reproduced the same pattern at the ensemble level, confirming that F1 reflects class bias as much as classification skill on this target.

The other important finding is **what the model's behavior reveals about the target**: per-fold analysis showed the SVM frequently collapsing to single-class predictions per 42-day window, and probability distributions for actual UP vs. actual DOWN days overlap almost entirely. This is consistent with daily equity direction being a noise-dominated signal that traditional ML on technical features cannot meaningfully separate. The features carry information — but not enough to push prediction beyond the ~52% boundary that decades of research have established for this problem.

A breakthrough would require categorically different signal: sentiment data, options market data (implied volatility surfaces, put/call ratios), order flow, or macroeconomic factors. These were out of scope for this project given the timeline and the decision to limit inputs to what is available through the Yahoo Finance API and derived features.

### Regression

This project evaluated four walk-forward configurations across two tickers (SPY, TSLA), and three model families (linear, ridge, tree-based) over a 10-year Yahoo Finance dataset (2015–2024). The progression from a 63-day training window with binary targets (Iteration 1) to refined targets and embargo (Iteration 2), and the further experiments with extreme short windows (Iterations 3 and 4), produced a clear and consistent picture. Following the presentation, additional optimizations and model testing done for both SPY and TSLA to see if further improvement can be done.

#### What the models can and cannot predict

**Realized volatility is learnable; raw return direction is not.** Switching the regression target from 5-day forward return (Iteration 1) to 5-day forward realized volatility (Iterations 2–4) was the single most impactful design decision. The volatility target is directly driven by the technical features already in the model — VIX, daily range, Bollinger Band position, and rolling volatility measures all carry genuine signal about near-term turbulence. Raw return direction is close to a random walk at short horizons; no configuration in any iteration produced positive overall R² for that target.

#### What the window experiments revealed

**Iteration 3 (3d/1d/0d)** produced the best raw numbers — lowest RMSE, all-positive seasonal R² for both tickers. These numbers are misleading: with 3 training samples and 1 test sample, models interpolate within a nearly static local window rather than generalizing. The extremely high R² values (SPY XGBoost Spring: +0.864) reflect local curve-fitting, not predictive power. The configuration is included as a cautionary example of how narrow windows can produce deceptively strong metrics.

**Iteration 4 (15d/5d/2d embargo)** is the most informative non-standard configuration. It is still underdetermined for linear models (15 samples × 16 features), but Ridge and XGBoost handle this robustly. SPY XGBoost achieves positive R² in Spring (+0.42) and Fall (+0.18) — the same seasons where Iteration 2 showed its best results — confirming these as genuinely learnable market periods rather than artifacts of one window size. The 5-day test window also produces stable fold-level metrics, making standard deviations meaningful.

**Iteration 2 (SPY: 189d/42d/5d embargo, TSLA: 59d/42d/5d embargo)** remains the best overall configuration. SPY's 189-day training window provides roughly 9 months of history per fold — enough for Ridge Regression to outperform XGBoost and achieve positive R² in three of four seasons (Spring +0.273, Fall +0.144, Winter +0.040). The 42-day test window produces stable fold-level metrics with meaningful standard deviations. It is the only configuration where SPY shows multi-season positive R², and the only configuration outside Iteration 3 where both tickers produce consistently interpretable, non-degenerate results across all models.

**Post-Presentation (SPY: 159d/42d/5d embargo, TSLA: 120d/42d/5d embargo)** expands the feature set from 16 to 26 by adding eight engineered features: the HAR-RV quarterly extension (rv_65d), four VIX-derived signals (vix_change, vix_lag_1, vix_lag_5, vix_ma20_ratio), return distribution shape features (skew_20, kurt_20), and a volatility regime indicator (vol_regime). For SPY, the additional features, increased training, plus leveraging the LightGBM model showed a net improvement. The performance improvement comparison between iteration2 and the post presentation run showed an improvement to from −0.058 (Iter2 XGBoost) to +0.363 (post-pres XGBoost), though there was a regression in the Spring performance going from +0.273 to ~0. Comparing XGBoost vs LightGBM, LightGBM has an improvement showing RMSE 0.005119, R² −0.848. With TSLA, there was a change to both the training window and the additional features which showed improvement. Updating the training window showed improvement to the R² score going from −1.4693 to −0.9454. Seasonally Spring R² improves by +0.338, Fall by +0.332, but summer does appear to be flat. Comparing LightGBM vs XGBoost has a trade-off where XGBoost wins on R² but LightGBM scores better with RMSE. However, the expansion of the features showed this had a negative impact where the R² decreases to −1.293 and LightGBM R² decreases to −1.316.   

#### The SPY/TSLA divide

Every metric confirms that SPY is more predictable than TSLA across all configurations. SPY's lower volatility, absence of idiosyncratic regime shifts, and cleaner alignment between technical indicators and price behavior makes regression meaningfully learnable in the right conditions. TSLA's meme-stock surge (2020–2022) and post-election spike (2024) introduce regime discontinuities that no 3–63 day training window can anticipate. The features that predict SPY volatility during normal market periods have little explanatory power when TSLA enters one of its structural breaks. Feature expansion showed a significant improvement to SPY whereas this showed degrading results for TSLA.

#### Model family conclusions

**XGBoost is the best regression model for TSLA across Iterations 1–4 and for SPY at shorter windows (Iter1, 3, 4).** Its combination of tree-based non-linearity and resistance to underdetermined conditions keeps it tractable when Linear Regression collapses catastrophically (R² = −3,582 for SPY, −257,532 for TSLA in Iteration 4). **For SPY at the 189-day window (Iter2), Ridge Regression surpasses XGBoost** (RMSE 0.005168 vs 0.005473; R² −0.991 vs −1.487): sufficient history allows L2 regularization to converge to stable coefficients that tree ensembles cannot exploit with equal efficiency. Linear Regression without regularization should not be used when training windows approach the feature count.

**In the Post-Presentation runs, LightGBM was introduced as an additional gradient boosting model** with leaf-wise tree growth, and a consistent finding held across both tickers: the original 16-feature set outperformed the expanded 26-feature configuration at the same training window. For SPY at the 159-day window, LightGBM with original features (RMSE 0.005119, R² −0.848) edges XGBoost (RMSE 0.005185, R² −0.909); the 8 expanded features added variance without improving generalization. For TSLA, the controlled comparison at the 120-day window confirms the same pattern: the original 16-feature set produced XGBoost RMSE 0.018337 (R² −0.945) and LightGBM RMSE 0.018177 (R² −0.983), while adding the expanded features degraded both models — XGBoost RMSE rose to 0.019192 (R² −1.293) and LightGBM to 0.019133 (R² −1.316). The VIX-derived signals measure broad-market fear rather than TSLA-specific volatility, and the distributional features (skew_20, kurt_20, vol_regime) introduced variance the 120-day window cannot regularize. For TSLA, increasing the training window is the effective lever; feature expansion beyond the original 16 is counterproductive.

#### Broader takeaway

Features created from price and volume data did provide enough of a signal for predicting near-term volatility for a stable ticker, but unfortunately was not able to replicate the same behavior for a more volatile one. That signal is consistent across model families and seasons (strongest in Fall) and holds up under walk-forward validation with embargo. The features were not strong enough to predict directional price movement — especially for high-volatility individual stocks. The results have reached a near ceiling given the features, the decision to limit the dataset to only metrics from the API and derived values limits the ability to further improve the models. Should this project move further forward, additional feature engineering could help improve the score and adding in features such as earnings, sentiment, and executive announcements could improve the prediction for TSLA.
