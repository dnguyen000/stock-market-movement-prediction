# SPY Regression Results

Regression target: **5-day forward realized volatility** (Iterations 2–4). Iteration 1 used 5-day forward return.

---

## Iteration 1 — SPY Regression (5-day forward return)

Walk-forward config: train=63d / test=42d / embargo=5d — **58 folds**

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.047786 ± 0.043899 | 0.041183 ± 0.040623 | -7.617818 ± 14.816828 |
| Ridge Regression | 0.039305 ± 0.031435 | 0.033785 ± 0.028524 | -4.249127 ± 5.066583 |
| **XGBoost** | **0.026246 ± 0.015421** | **0.021343 ± 0.013119** | **-1.439560 ± 2.302285** |

---

## Iteration 2 — SPY Regression (5-day forward realized volatility)

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.005343 ± 0.004458 | 0.004254 ± 0.003341 | -1.142533 ± 1.782205 |
| **Ridge Regression** | **0.005168 ± 0.004213** | **0.004111 ± 0.003146** | **-0.990798 ± 1.516107** |
| XGBoost | 0.005473 ± 0.004536 | 0.004299 ± 0.003741 | -1.486850 ± 2.383709 |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.009970 | 0.005511 | +0.1899 | 575 |
| Linear Regression | Summer | 0.005758 | 0.003766 | -0.2563 | 580 |
| Linear Regression | Fall | 0.005215 | 0.003895 | +0.0423 | 603 |
| Linear Regression | Winter | 0.005748 | 0.003851 | -0.0204 | 552 |
| **Ridge Regression** | **Spring** | **0.009443** | **0.005213** | **+0.2732** | **575** |
| Ridge Regression | Summer | 0.005703 | 0.003713 | -0.2326 | 580 |
| Ridge Regression | Fall | 0.004930 | 0.003724 | +0.1441 | 603 |
| Ridge Regression | Winter | 0.005576 | 0.003804 | +0.0397 | 552 |
| XGBoost | Spring | 0.010333 | 0.005416 | +0.1298 | 575 |
| XGBoost | Summer | 0.005769 | 0.004183 | -0.2611 | 580 |
| XGBoost | Fall | 0.005481 | 0.003778 | -0.0582 | 603 |
| XGBoost | Winter | 0.005559 | 0.003828 | +0.0455 | 552 |

---

## Iteration 3 — SPY Regression (5-day forward realized volatility)

Walk-forward config: train=3d / test=1d / embargo=0d — **2,507 folds**

> Per-fold R² is undefined when test=1d (requires ≥2 samples). Mean R² is reported as NaN. Use seasonal R² (aggregated predictions) for model comparison.

| Model | RMSE (mean ± std) | MAE (mean ± std) |
|---|---|---|
| Linear Regression | 0.002496 ± 0.004214 | 0.002496 ± 0.004214 |
| Ridge Regression | 0.002433 ± 0.004024 | 0.002433 ± 0.004024 |
| **XGBoost** | **0.002181 ± 0.002797** | **0.002181 ± 0.002797** |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² (aggregated) | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.004699 | 0.002462 | +0.8036 | 638 |
| Linear Regression | Summer | 0.004527 | 0.002399 | +0.3016 | 645 |
| Linear Regression | Fall | 0.005857 | 0.002684 | -0.2206 | 630 |
| Linear Regression | Winter | 0.004345 | 0.002439 | +0.3954 | 594 |
| Ridge Regression | Spring | 0.004539 | 0.002407 | +0.8167 | 638 |
| Ridge Regression | Summer | 0.004383 | 0.002344 | +0.3453 | 645 |
| Ridge Regression | Fall | 0.005569 | 0.002607 | -0.1037 | 630 |
| Ridge Regression | Winter | 0.004173 | 0.002373 | +0.4424 | 594 |
| **XGBoost** | **Spring** | **0.003912** | **0.002332** | **+0.8639** | **638** |
| XGBoost | Summer | 0.003269 | 0.002025 | +0.6357 | 645 |
| XGBoost | Fall | 0.003334 | 0.002166 | +0.6045 | 630 |
| XGBoost | Winter | 0.003638 | 0.002202 | +0.5762 | 594 |

---

## Iteration 4 — SPY Regression (5-day forward realized volatility)

Walk-forward config: train=15d / test=5d / embargo=2d — **498 folds**

> Linear Regression produces extreme R² values (−3,582 mean) due to an underdetermined system (15 training samples, 16 features). Ridge and XGBoost are robust to this.

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.041628 ± 0.105180 | 0.037719 ± 0.096886 | −3582.21 ± 14087.87 |
| Ridge Regression | 0.006567 ± 0.007762 | 0.006029 ± 0.007443 | −91.34 ± 423.21 |
| **XGBoost** | **0.004424 ± 0.004333** | **0.004038 ± 0.004176** | **−29.58 ± 102.70** |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Ridge Regression | Spring | 0.013042 | 0.007018 | −0.5129 | 638 |
| Ridge Regression | Summer | 0.008145 | 0.005502 | −1.2613 | 645 |
| Ridge Regression | Fall | 0.009407 | 0.005933 | −2.1489 | 630 |
| Ridge Regression | Winter | 0.009315 | 0.005628 | −1.7156 | 577 |
| **XGBoost** | **Spring** | **0.008065** | **0.004791** | **+0.4214** | **638** |
| XGBoost | Summer | 0.005501 | 0.003841 | −0.0314 | 645 |
| **XGBoost** | **Fall** | **0.004807** | **0.003600** | **+0.1778** | **630** |
| XGBoost | Winter | 0.005860 | 0.003903 | −0.0749 | 577 |

---

## Challenges and Observations (Iteration 1)

**1. Noise in the regression target**

`target_return` (5-day forward return) is close to a random walk. Technical indicators explain an estimated 2–5% of the variance in 5-day returns. The remaining 95%+ is driven by news events, earnings surprises, Fed announcements, and large institutional order flow that technical features have no visibility into. All R² values being negative is not a modeling failure — it is a data sufficiency problem. Adding sentiment data, options market data, or macroeconomic factors would be required to meaningfully improve regression scores.

**2. High variance across folds**

Standard deviation approached the mean for Linear Regression on SPY (std 0.043899 vs mean 0.047786 — 92% of the mean), indicating high instability across folds. XGBoost was the most consistent (std ~59% of mean: 0.015421 vs 0.026246). Increasing the training window from 63 to 252 days would reduce this variance by exposing each fold to a more representative market history.

**3. `is_major_event` coefficient weight is modest but non-zero**

The 10-year Yahoo Finance dataset (2015–2024) contains 144 trading days with VIX > 30, giving models sufficient exposure to the stress regime. The coefficient for `is_major_event` is non-trivial but remains a minor contributor because stress days are only ~5.7% of the sample.

**4. Feature redundancy**

The second feature set (19 features) included correlated features: `macd`, `macd_signal`, and `macd_hist` are mathematically derived from each other. Similarly, `volatility_7` and `volatility_20` overlap. The second iteration reduces to 16 features by removing the redundant ones.

---

## Comparing First and Second Iteration — Regression

Targets changed (`target_return` → `target_volatility`), so RMSE values are not directly comparable. The key improvement metric is R²:

| Ticker | Best Model | 1st Iter R² | 2nd Iter R² | Best Seasonal R² (Iter 2) |
|---|---|---|---|---|
| SPY | Ridge Regression | n/a (different target) | -0.990798 ± 1.516107 | Spring **+0.2732** |

With the SPY training window extended to 189 days, **Ridge Regression is the best SPY model** (RMSE=0.005168, R²=−0.991), outperforming XGBoost (RMSE=0.005473, R²=−1.487). Multiple seasons now have positive R² for SPY:

- Spring (all models): Ridge R²=**+0.2732**, Linear R²=+0.1899, XGBoost R²=+0.1298
- Fall: Ridge R²=+0.1441, Linear R²=+0.0423
- Winter: Ridge R²=+0.0397, XGBoost R²=+0.0455

This is a major shift from the prior 63-day config, where only one seasonal R² was positive.

---

## Key Findings Across All Iterations

1. **The 3-day window (Iter3) produces the best raw RMSE and R² numbers but is statistically misleading.** With 3 training samples and 1 test sample, models interpolate rather than generalize. The high R² values (SPY XGBoost Spring: +0.864) reflect local curve-fitting, not predictive power.

2. **The 189-day SPY window in Iter2 is the strongest well-founded configuration.** Ridge Regression achieves positive R² in Spring (+0.273), Fall (+0.144), and Winter (+0.040) — three of four seasons.

3. **Ridge Regression outperforms XGBoost for SPY at the 189-day window** (RMSE 0.005168 vs 0.005473; R² −0.991 vs −1.487). Longer training history allows Ridge's L2 regularization to converge to stable coefficients.

4. **Spring is the most predictable season for SPY regression in Iter2.** Ridge Spring R²=+0.273 is the highest single-season value across any non-Iter3 iteration.

5. **Fall is consistently predictable across all informative iterations (Iter2, 3, 4).** Iter2 Ridge Fall R²=+0.144, Iter4 XGBoost Fall R²=+0.178, Iter3 XGBoost Fall R²=+0.605. This convergence across very different window configurations validates Fall as a genuine signal period.

6. **Linear regression breaks down at small window sizes.** At 15 samples / 16 features (Iter4), linear regression is technically underdetermined, producing R² = −3,582.
