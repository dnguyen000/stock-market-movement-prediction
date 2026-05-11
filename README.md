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
  - Automate documentation of results and clean summarizations of prompted notes taken of model results
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

## How Testing Is Done

Testing uses **walk-forward validation** — a time-series aware evaluation strategy that prevents data leakage and simulates realistic trading conditions.

### Window Configurations Across Iterations

| Iteration | Train | Test | Embargo | Approx. Folds | Dataset |
|---|---|---|---|---|---|
| 1 | 63d | 42d | 5d | ~58 | Yahoo Finance, 10yr |
| **2** | **189d (SPY) / 59d (TSLA)** | **42d** | **5d** | **55 (SPY) / 58 (TSLA)** | **Yahoo Finance, 10yr** |
| 3 | 3d | 1d | 0d | ~2,507 | Yahoo Finance, 10yr |
| 4 | 15d | 5d | 2d | ~498 | Yahoo Finance, 10yr |

Iteration 2 is the primary production configuration. Iterations 3 and 4 explore the effect of shorter windows and are included as cautionary comparisons.

> **Note on scope:** All classification experiments were conducted under Iteration 2 only (SPY 189d / TSLA 59d, 42d test, 5d embargo). Iterations 1, 3, and 4 apply to the regression experiments — see the Regression section.

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

## Models

### Classification

All classification models use a shared feature set of **19 engineered features** derived from price, volume, momentum, volatility, and market-stress signals. Each model is evaluated with walk-forward validation using `class_weight='balanced'` to mitigate the natural imbalance between UP and DOWN days. Features are standardized via `StandardScaler` fit inside each training fold only (the scaler is never fit on test data). All models use `random_state=42` for reproducibility. SVM additionally enables `probability=True` for probability calibration.

Standard configuration: train=189d / test=42d / embargo=5d (SPY: 55 folds; TSLA: train=59d / 58 folds, except Random Forest which uses 189d/55 folds for both tickers).

#### Models
Predict the **next-day price direction** — whether the stock's closing price will be higher (UP) or lower (DOWN) than the prior day's close.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Logistic Regression** | Linear baseline classifier. Fits a logistic function to the feature space to output a probability of UP. L1 and L2 regularization are compared via grid search. Provides interpretable coefficients but is limited to linear decision boundaries. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Random Forest** | Bagging ensemble of decision trees. Each tree is trained on a bootstrap sample of the training window; majority vote across trees determines the prediction. Captures non-linear interactions between features without requiring explicit kernel specification. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Support Vector Machine (SVM)** | Kernel-based classifier using an RBF (radial basis function) kernel. Finds the maximum-margin hyperplane in a high-dimensional feature space, allowing it to capture non-linear boundaries. Regularization strength (C) and kernel width (gamma) are tuned via grid search. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **XGBoost** | Gradient boosting ensemble that builds decision trees sequentially, each correcting the residual errors of the prior tree. Uses `scale_pos_weight` to handle class imbalance between UP and DOWN days. Provides feature importance scores ranking which technical indicators most influence directional predictions. | F1 Score | Accuracy, Precision, Recall, Specificity |
| **Voting Ensemble (Hard Vote)** | Combines all four classifiers above using a majority vote rule. Two voting thresholds are evaluated: a loose rule (≥ 2 of 4 models predict UP) and a strict rule (≥ 3 of 4 models predict UP). Component configurations are chosen to maximize vote diversity rather than individual F1, providing more robust and consistent predictions than any single model. | F1 Score | Accuracy, Precision, Recall, Specificity |

**Exploratory variants.** Four notebooks (`rf-Copy1_with_17_features.ipynb`, `rf-tsla-Copy1_with_17_features.ipynb`, `xboost_spy-Copy1_witth_17_features.ipynb`, `xboost_tsla-Copy1_with_17_features.ipynb`) test reduced 16-feature variants (dropping `month`, `is_earnings_week`, `is_major_event`) to assess whether trimming low-importance features improves performance. Results were within fold-level noise of the canonical 19-feature models, so the trimmed variants were not used in final reporting. These notebooks are retained for reference but are not part of the production pipeline.


### Regression

All models are evaluated using walk-forward validation across both tickers independently.

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

Component models: SVM (C=100, gamma=0.1) + Logistic Regression (C=0.001, l2) + Random Forest (n_est=200, depth=6, min_leaf=5, sqrt) + XGBoost (n_est=500, depth=4, lr=0.01, subs=0.8)

We tested two majority-voting rules to evaluate how the decision threshold affects class balance:

- **Loose majority (≥ 2 of 4 models predict UP)** — predicts UP whenever any two models agree
- **Strict majority (≥ 3 of 4 models predict UP)** — predicts UP only with broader consensus

| Metric | Loose (≥ 2 of 4) | Strict (≥ 3 of 4) |
|---|---|---|
| Accuracy | 0.521 ± 0.091 | 0.504 ± 0.079 |
| F1 (UP) | 0.574 ± 0.119 | 0.478 ± 0.152 |
| Precision (UP) | 0.567 | 0.568 |
| Recall (UP) | 0.628 | 0.463 |
| Specificity (DOWN) | 0.397 | 0.560 |
| Macro F1 | 0.51 | 0.50 |

**Interpretation.** The loose rule produces a higher F1 but achieves it by predicting UP more often — catching more actual UP days (recall 0.628) at the cost of missing DOWN days (specificity 0.397). The strict rule is the mirror image: predicts UP less often, catches more DOWN days (specificity 0.560), at the cost of UP recall (0.463). Precision is essentially identical (0.567 vs 0.568) — when either rule says UP, it is correct equally often. Macro-averaged F1 (which treats both classes equally) is functionally identical between the two rules (~0.50), confirming that the two configurations sit at the same overall predictive ceiling and differ only in **which class absorbs the errors**.

This is a concrete example of why single-class F1 is misleading on imbalanced targets: the same underlying model can produce F1 of 0.574 or 0.478 depending on a single threshold choice, without any change in actual predictive ability. We retain the loose rule (≥ 2) as our reported "best F1" configuration for consistency with the primary metric, but the strict rule is the more honest representation of the model's balanced classification capability.

**Seasonal breakdown (loose rule, ≥ 2 of 4):**

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 575 | 0.577 |
| Summer | 580 | 0.605 |
| Fall | 603 | 0.587 |
| Winter | 552 | 0.591 |

#### Best Model — TSLA: Voting Ensemble (Hard Vote)

Component models: SVM (C=10, gamma=0.1) + Logistic Regression (C=10, l1) + Random Forest (n_est=200, depth=4, min_leaf=10, sqrt) + XGBoost (n_est=100, depth=3, lr=0.01, subs=1.0)

| Metric | Value |
|---|---|
| F1 (UP) | 0.545 ± 0.153 |
| Accuracy | 0.502 ± 0.081 |
| Precision (UP) | 0.532 |
| Recall / Sensitivity (UP) | 0.653 |
| Specificity (Recall DOWN) | 0.361 |

**Seasonal breakdown:**

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 613 | 0.534 |
| Summer | 645 | 0.549 |
| Fall | 630 | 0.608 |
| Winter | 548 | 0.600 |

> **Note on class balance.** TSLA shows the same asymmetric pattern as SPY but more pronounced: UP recall 0.653, DOWN recall 0.361. Per-fold analysis (visible in `svm_tsla.ipynb`) shows the SVM frequently collapses to predicting a single class for an entire 42-day test window — alternating between "predict UP for all 42 days" and "predict DOWN for all 42 days" depending on the training slice. Probability distributions for actual UP vs. actual DOWN days overlap almost entirely (mean P(UP) = 0.523 on UP days, 0.519 on DOWN days), indicating the model cannot reliably distinguish the two classes from the available technical features. The reported F1 reflects this bias and should not be read as evidence of balanced classification ability.

### Key Findings Across Classification Models

1. **F1 alone is misleading on this target.** Across all four base models and both tickers, the highest-F1 configurations achieve their scores by biasing predictions toward the UP class. The clearest demonstration is the voting-rule comparison on SPY: switching the ensemble from ≥ 2 of 4 to ≥ 3 of 4 changes F1 from 0.574 to 0.478, recall from 0.628 to 0.463, and specificity from 0.397 to 0.560 — while macro F1 (~0.50) and precision (0.567) remain essentially unchanged. The same underlying model produces dramatically different F1 numbers depending on a single threshold, confirming that F1 reflects class bias as much as it reflects classification ability on this target. Grid search results for individual models showed the same pattern: SPY SVM's best F1 (0.61) came from a configuration with specificity 0.19, while macro F1 was actually lower than the baseline.

2. **SPY is more predictable than TSLA but only marginally.** SPY accuracy 0.52 vs TSLA accuracy 0.50. The gap is real but small — well within published ceilings for daily direction prediction on liquid equities (52–55%).

3. **Per-fold variance dominates the picture.** F1 standard deviations of ±0.12 (SPY) and ±0.15 (TSLA) across folds are large relative to the mean F1 itself. Individual folds range from near-zero F1 to F1 above 0.75. This reflects market regime shifts more than model instability — the same model produces very different results in trending vs. choppy market periods.

4. **Seasonal performance is consistent.** F1 across Spring/Summer/Fall/Winter is within 0.03 for SPY and 0.07 for TSLA, indicating no strong seasonal effect dominates the model's behavior. Differences are within fold-level noise.

5. **Methodology limitation — hyperparameter selection bias.** Grid search for each base model was conducted across the same walk-forward folds later used for final evaluation. The best hyperparameter configuration was selected based on its mean F1 across those folds. This is a form of selection bias: the reported best-model metrics are optimistic relative to what an unbiased held-out test would show. A nested cross-validation scheme (selecting hyperparameters on inner folds, evaluating on outer folds) would address this but was not implemented due to scope constraints. The magnitude of this bias is likely modest — most grid configurations produced F1 within 0.02–0.05 of each other on a given ticker, suggesting model performance is not highly sensitive to specific hyperparameter choices in this region of the search space.

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

The most important finding from the classification track is not the F1 number itself but **what the model's behavior reveals about the target**: per-fold analysis showed the SVM frequently collapsing to single-class predictions per 42-day window, and probability distributions for actual UP vs. actual DOWN days overlap almost entirely. This is consistent with daily equity direction being a noise-dominated signal that traditional ML on technical features cannot meaningfully separate. The features carry information — but not enough to push prediction beyond the ~52% boundary that decades of research have established for this problem.

A breakthrough would require categorically different signal: sentiment data, options market data (implied volatility surfaces, put/call ratios), order flow, or macroeconomic factors. These were out of scope for this project given the timeline and the decision to limit inputs to what is available through the Yahoo Finance API and derived features.

### Regression

This project evaluated four walk-forward configurations across two tickers (SPY, TSLA), two tasks (regression, classification), and three model families (linear, ridge, tree-based) over a 10-year Yahoo Finance dataset (2015–2024). The progression from a 63-day training window with binary targets (Iteration 1) to refined targets and embargo (Iteration 2), and the further experiments with extreme short windows (Iterations 3 and 4), produced a clear and consistent picture.

#### What the models can and cannot predict

**Realized volatility is learnable; raw return direction is not.** Switching the regression target from 5-day forward return (Iteration 1) to 5-day forward realized volatility (Iterations 2–4) was the single most impactful design decision. The volatility target is directly driven by the technical features already in the model — VIX, daily range, Bollinger Band position, and rolling volatility measures all carry genuine signal about near-term turbulence. Raw return direction is close to a random walk at short horizons; no configuration in any iteration produced positive overall R² for that target.

#### What the window experiments revealed

**Iteration 3 (3d/1d/0d)** produced the best raw numbers — lowest RMSE, all-positive seasonal R² for both tickers. These numbers are misleading: with 3 training samples and 1 test sample, models interpolate within a nearly static local window rather than generalizing. The extremely high R² values (SPY XGBoost Spring: +0.864) reflect local curve-fitting, not predictive power. The configuration is included as a cautionary example of how narrow windows can produce deceptively strong metrics.

**Iteration 4 (15d/5d/2d embargo)** is the most informative non-standard configuration. It is still underdetermined for linear models (15 samples × 16 features), but Ridge and XGBoost handle this robustly. SPY XGBoost achieves positive R² in Spring (+0.42) and Fall (+0.18) — the same seasons where Iteration 2 showed its best results — confirming these as genuinely learnable market periods rather than artifacts of one window size. The 5-day test window also produces stable fold-level metrics, making standard deviations meaningful.

**Iteration 2 (SPY: 189d/42d/5d embargo, TSLA: 59d/42d/5d embargo)** remains the best overall configuration. SPY's 189-day training window provides roughly 9 months of history per fold — enough for Ridge Regression to outperform XGBoost and achieve positive R² in three of four seasons (Spring +0.273, Fall +0.144, Winter +0.040). The 42-day test window produces stable fold-level metrics with meaningful standard deviations. It is the only configuration where SPY shows multi-season positive R², and the only configuration outside Iteration 3 where both tickers produce consistently interpretable, non-degenerate results across all models.

**Post-Presentation (SPY: 159d/42d/5d embargo, TSLA: 120d/42d/5d embargo)** expands the feature set from 18 to 26 by adding eight engineered features: the HAR-RV quarterly extension (rv_65d), four VIX-derived signals (vix_change, vix_lag_1, vix_lag_5, vix_ma20_ratio), return distribution shape features (skew_20, kurt_20), and a volatility regime indicator (vol_regime). Results diverge sharply by ticker. For SPY, the additions are a net improvement but was counterproductive for TSLA

#### The SPY/TSLA divide

Every metric confirms that SPY is more predictable than TSLA across all configurations. SPY's lower volatility, absence of idiosyncratic regime shifts, and cleaner alignment between technical indicators and price behavior makes regression meaningfully learnable in the right conditions. TSLA's meme-stock surge (2020–2022) and post-election spike (2024) introduce regime discontinuities that no 3–63 day training window can anticipate. The features that predict SPY volatility during normal market periods have little explanatory power when TSLA enters one of its structural breaks.

#### Model family conclusions

**XGBoost is the best regression model for TSLA in every iteration and for SPY at shorter windows (Iter1, 3, 4).** Its combination of tree-based non-linearity and resistance to underdetermined conditions keeps it tractable when Linear Regression collapses catastrophically (R² = −3,582 for SPY, −257,532 for TSLA in Iteration 4). **For SPY at the 189-day window (Iter2), Ridge Regression surpasses XGBoost** (RMSE 0.005168 vs 0.005473; R² −0.991 vs −1.487): sufficient history allows L2 regularization to learn stable patterns that tree ensembles cannot exploit with equal efficiency. Linear Regression without regularization should not be used when training windows approach the feature count.

#### Broader takeaway

Features created from price and volume data did provide enough of a signal for predicting near-term volatility for a stable ticker, but unfortunately was not able to replicate the same behavior for a more volatile one. That signal is consistent across model families and seasons (strongest in Fall) and holds up under walk-forward validation with embargo. The features were not strong enough to predict directional price movement — especially for high-volatility individual stocks. The results have reached a near ceiling given the features, the decision to limit the dataset to only metrics from the API and derived values limits the ability to further improve the models. Should this project move further forward, additional feature engineering could help improve the score and adding in features such as earnings, sentiment, and executive announcements could improve the prediction for TSLA.
