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

### Machine Learning Terms

| Term | Definition |
|---|---|
| **Feature (Independent Variable)** | An input variable used by the model to make a prediction. In this project, features include daily return, RSI, volume change, etc. |
| **Target (Dependent Variable)** | The value the model is trying to predict. Here: next-day direction (classification) or 5-day price change (regression). |
| **Classification** | A type of ML task where the model predicts a discrete category. Here: UP or DOWN. |
| **Regression** | A type of ML task where the model predicts a continuous numeric value. Here: 5-day forward return. |
| **Logistic Regression** | A classification algorithm that estimates the probability of a binary outcome (UP or DOWN) using a logistic curve. Simple and interpretable. |
| **Linear Regression** | A regression algorithm that fits a straight line through training data to predict continuous values. |
| **Ridge Regression** | A variant of linear regression that adds a penalty for large coefficients, reducing overfitting when features are correlated. |
| **Random Forest** | An ensemble of many decision trees that each vote on the prediction. More powerful than a single tree and provides built-in feature importance scores. |
| **XGBoost** | A gradient boosting algorithm that builds trees sequentially, each correcting the errors of the previous. Often the top performer on structured/tabular data. |
| **Training** | The process of fitting a model to historical data so it learns patterns. |
| **Testing** | Evaluating the model on data it has never seen to measure real-world performance. |
| **Overfitting** | When a model learns the training data too precisely, including its noise, and fails to generalize to new data. |
| **Walk-Forward Validation** | A time-aware testing strategy where the model is trained on a past window of data and tested on the immediately following window, repeating across multiple periods. Prevents data leakage and simulates real trading conditions. |
| **Data Leakage** | When information from the future accidentally influences the training process, producing misleadingly optimistic results. |
| **Feature Importance / Weights** | A score assigned to each feature indicating how much it influenced the model's predictions. Computed via `model.coef_` for linear models and `model.feature_importances_` for tree-based models. |
| **Standardization (StandardScaler)** | Rescaling features to have a mean of 0 and standard deviation of 1, so no single feature dominates due to scale differences. |
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and actual values. Intuitive and robust to outliers. |
| **RMSE (Root Mean Squared Error)** | Square root of the average squared error. Penalizes large mistakes more than MAE — important in finance where big misses are costly. |
| **R² (R-squared)** | Proportion of variance in the target explained by the model. A value of 1.0 is perfect; negative values mean the model performs worse than simply predicting the mean. |
| **Precision** | Of all days the model predicted as UP, the fraction that actually went UP. Low precision means the model generates many false buy signals — a trader acting on every UP prediction would frequently lose money on days that actually went down. |
| **Recall** | Of all days that actually went UP, the fraction the model correctly identified. Low recall means the model misses many genuine opportunities — a conservative model that rarely predicts UP will have high precision but low recall. |
| **F1 Score** | Harmonic mean of precision and recall. A single number that balances both risks: false buy signals (low precision) and missed opportunities (low recall). More informative than accuracy alone when classes are nearly balanced. |
| **Confusion Matrix** | A table showing correct and incorrect predictions broken down by class (e.g., Predicted Up vs. Actual Up). |
| **Seasonal Analysis** | Grouping model results by time of year (spring, summer, fall, winter) to identify whether market conditions in certain seasons are more predictable. |
| **Realized Volatility** | The actual, observed volatility of a stock over a past window — computed as the standard deviation of daily returns over N days. Used as the regression target in the second iteration (`target_volatility`), replacing the 5-day forward return. Volatility is more learnable than raw return direction because it is directly driven by the technical features already in the model (VIX, daily range, RSI). |
| **Softened Target** | A classification target that filters out small, ambiguous price moves rather than forcing a binary UP/DOWN label on every day. In the second iteration, days with returns within ±0.5% (SPY) or ±1.5% (TSLA) of zero are labeled FLAT (0) instead of UP or DOWN. This gives the model cleaner signal on days where a genuine trend is present and reduces noise from microstructure. |
| **FLAT Class** | The third label in the softened 3-class classification target (Down = -1, Flat = 0, Up = 1). Represents a trading day where the stock's move was too small to be a meaningful directional signal. Adding FLAT lowers the random-classifier F1 baseline from ~0.50 to ~0.33, so scores above ~0.38 represent meaningful learning. |
| **L2 Regularization** | A penalty added to a model's loss function that discourages large coefficient values. Proportional to the sum of squared coefficients. Ridge Regression applies L2 regularization to reduce overfitting when input features are correlated with each other — which is common with technical indicators (e.g., RSI, Bollinger Band position, and distance from MA all measure similar things). |
| **Embargo (Walk-Forward)** | A mandatory gap of N trading days inserted between the end of a training window and the start of the test window in walk-forward validation. Prevents feature leakage when features are computed over a rolling window — without an embargo, the last rows of the training window and the first rows of the test window share overlapping feature calculations. In this project a **5-day embargo** is applied at each fold boundary. |

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

### Process
Each fold trains the model exclusively on its training window, then predicts the immediately following test window. An embargo separates the two to prevent feature leakage from rolling window calculations. The next fold begins immediately after the previous test window ends:

```
Iteration 2 example (63d/21d/5d):
Fold 1: Train [day 1   → day 63],  Embargo [day 64–68],  Test [day 69  → day 89]
Fold 2: Train [day 69  → day 131], Embargo [day 132–136], Test [day 137 → day 157]
...

Iteration 4 example (15d/5d/2d):
Fold 1: Train [day 1  → day 15],  Embargo [day 16–17],  Test [day 18 → day 22]
Fold 2: Train [day 6  → day 20],  Embargo [day 21–22],  Test [day 23 → day 27]
...
```

This approach captures performance across multiple market conditions rather than a single arbitrary split.

### Metrics Reported
Results are aggregated as **mean ± standard deviation** across all folds.

| Task | Primary Metric | Supporting Metrics |
|---|---|---|
| Regression | RMSE | MAE, R² |
| Classification | F1 Score | Accuracy, Confusion Matrix |

Standard deviation across folds measures **consistency** — a model with lower std is more reliable across different market conditions.

---

## Models

All models are evaluated using walk-forward validation across both tickers independently.

### Regression Models
Predict the **5-day forward return** — the percentage price change from today's close to the close 5 trading days from now. A positive value means the model expects the price to rise; negative means it expects a decline.

| Model | Purpose | Primary Metric | Supporting Metrics |
|---|---|---|---|
| **Linear Regression** | Baseline model. Fits a straight line through the feature space to predict the 5-day return as a weighted sum of inputs. Coefficients reveal which features most influence the prediction. | RMSE | MAE, R² |
| **Ridge Regression** | Extension of linear regression with L2 regularization — penalizes large coefficients to reduce overfitting. Particularly useful here because technical indicators (RSI, Bollinger Band position, distance from MA) are often correlated with each other. | RMSE | MAE, R² |
| **XGBoost** | Gradient boosting ensemble that builds decision trees sequentially, each correcting the residual errors of the prior tree. Captures non-linear relationships between features and returns that linear models cannot, and provides feature importance scores. | RMSE | MAE, R² |
---

## Results

Results are reported as **mean ± standard deviation** across all walk-forward folds. For granular per-iteration tables and seasonal breakdowns, see:
- [SPY Regression Results](results/spy_regression_results.md)
- [TSLA Regression Results](results/tsla_regression_results.md)

### Regression — SPY (best model per iteration)

| Iteration | Train Window | Best Model | RMSE (mean ± std) | Seasonal R² highlights |
|---|---|---|---|---|
| 1 (5-day return target) | 63d | XGBoost | 0.0262 ± 0.0154 | n/a (different target) |
| **2** | **189d** | **Ridge** | **0.005168 ± 0.004213** | Spring **+0.273**, Fall +0.144, Winter +0.040 |
| 3 | 3d | XGBoost | 0.002181 ± 0.002797 | All seasons positive (interpolation artifact) |
| 4 | 15d | XGBoost | 0.004424 ± 0.004333 | Spring **+0.421**, Fall **+0.178** |

### Regression — TSLA (best model per iteration)

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

This project evaluated four walk-forward configurations across two tickers (SPY, TSLA), two tasks (regression, classification), and three model families (linear, ridge, tree-based) over a 10-year Yahoo Finance dataset (2015–2024). The progression from a 63-day training window with binary targets (Iteration 1) to refined targets and embargo (Iteration 2), and the further experiments with extreme short windows (Iterations 3 and 4), produced a clear and consistent picture.

### What the models can and cannot predict

**Realized volatility is learnable; raw return direction is not.** Switching the regression target from 5-day forward return (Iteration 1) to 5-day forward realized volatility (Iterations 2–4) was the single most impactful design decision. The volatility target is directly driven by the technical features already in the model — VIX, daily range, Bollinger Band position, and rolling volatility measures all carry genuine signal about near-term turbulence. Raw return direction is close to a random walk at short horizons; no configuration in any iteration produced positive overall R² for that target.

### What the window experiments revealed

**Iteration 3 (3d/1d/0d)** produced the best raw numbers — lowest RMSE, all-positive seasonal R² for both tickers. These numbers are misleading: with 3 training samples and 1 test sample, models interpolate within a nearly static local window rather than generalizing. The extremely high R² values (SPY XGBoost Spring: +0.864) reflect local curve-fitting, not predictive power. The configuration is included as a cautionary example of how narrow windows can produce deceptively strong metrics.

**Iteration 4 (15d/5d/2d embargo)** is the most informative non-standard configuration. It is still underdetermined for linear models (15 samples × 16 features), but Ridge and XGBoost handle this robustly. SPY XGBoost achieves positive R² in Spring (+0.42) and Fall (+0.18) — the same seasons where Iteration 2 showed its best results — confirming these as genuinely learnable market periods rather than artifacts of one window size. The 5-day test window also produces stable fold-level metrics, making standard deviations meaningful.

**Iteration 2 (SPY: 189d/42d/5d embargo, TSLA: 59d/42d/5d embargo)** remains the best overall configuration. SPY's 189-day training window provides roughly 9 months of history per fold — enough for Ridge Regression to outperform XGBoost and achieve positive R² in three of four seasons (Spring +0.273, Fall +0.144, Winter +0.040). The 42-day test window produces stable fold-level metrics with meaningful standard deviations. It is the only configuration where SPY shows multi-season positive R², and the only configuration outside Iteration 3 where both tickers produce consistently interpretable, non-degenerate results across all models.

### The SPY/TSLA divide

Every metric confirms that SPY is more predictable than TSLA across all configurations. SPY's lower volatility, absence of idiosyncratic regime shifts, and cleaner alignment between technical indicators and price behavior makes regression meaningfully learnable in the right conditions. TSLA's meme-stock surge (2020–2022) and post-election spike (2024) introduce regime discontinuities that no 3–63 day training window can anticipate. The features that predict SPY volatility during normal market periods have little explanatory power when TSLA enters one of its structural breaks.

### Model family conclusions

**XGBoost is the best regression model for TSLA in every iteration and for SPY at shorter windows (Iter1, 3, 4).** Its combination of tree-based non-linearity and resistance to underdetermined conditions keeps it tractable when Linear Regression collapses catastrophically (R² = −3,582 for SPY, −257,532 for TSLA in Iteration 4). **For SPY at the 189-day window (Iter2), Ridge Regression surpasses XGBoost** (RMSE 0.005168 vs 0.005473; R² −0.991 vs −1.487): sufficient history allows L2 regularization to learn stable patterns that tree ensembles cannot exploit with equal efficiency. Linear Regression without regularization should not be used when training windows approach the feature count.

### Broader takeaway

Technical indicators derived from price and volume data provide a genuine but modest signal for predicting near-term volatility in stable, large-cap benchmarks. That signal is consistent across model families and seasons (strongest in Fall) and holds up under rigorous walk-forward validation with embargo. It is not sufficient to reliably predict directional price movement — especially for high-volatility individual stocks. 

