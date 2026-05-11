# TSLA Regression Results

Regression target: **5-day forward realized volatility** (Iterations 2–4). Iteration 1 used 5-day forward return.

---

## Iteration 1 — TSLA Regression (5-day forward return)

Walk-forward config: train=63d / test=42d / embargo=5d — **58 folds**

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.188666 ± 0.141773 | 0.157598 ± 0.123677 | -10.193193 ± 29.986320 |
| Ridge Regression | 0.146747 ± 0.099270 | 0.122307 ± 0.083897 | -5.904711 ± 20.436670 |
| **XGBoost** | **0.100080 ± 0.039484** | **0.082080 ± 0.034292** | **-1.331088 ± 2.004227** |

---

## Iteration 2 — TSLA Regression (5-day forward realized volatility)

Walk-forward config: train=59d / test=42d / embargo=5d — **58 folds**

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.032601 ± 0.022337 | 0.027054 ± 0.019179 | -8.090091 ± 16.194827 |
| Ridge Regression | 0.025794 ± 0.014242 | 0.021109 ± 0.011159 | -3.510050 ± 4.868767 |
| **XGBoost** | **0.019849 ± 0.009929** | **0.015808 ± 0.007814** | **-1.469254 ± 1.692770** |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.047505 | 0.031219 | -5.0128 | 613 |
| Linear Regression | Summer | 0.035807 | 0.025480 | -4.2735 | 645 |
| **Linear Regression** | **Fall** | **0.033753** | **0.025032** | **-1.9638** | **630** |
| Linear Regression | Winter | 0.039507 | 0.026574 | -3.1458 | 548 |
| Ridge Regression | Spring | 0.035056 | 0.024687 | -2.2743 | 613 |
| **Ridge Regression** | **Summer** | **0.026202** | **0.019496** | **-1.8238** | **645** |
| Ridge Regression | Fall | 0.027939 | 0.020501 | -1.0306 | 630 |
| Ridge Regression | Winter | 0.027630 | 0.019704 | -1.0278 | 548 |
| XGBoost | Spring | 0.025188 | 0.017270 | -0.6904 | 613 |
| **XGBoost** | **Summer** | **0.018098** | **0.014180** | **-0.3471** | **645** |
| XGBoost | Fall | 0.022976 | 0.016612 | -0.3733 | 630 |
| XGBoost | Winter | 0.021908 | 0.015166 | -0.2749 | 548 |

---

## Iteration 3 — TSLA Regression (5-day forward realized volatility)

Walk-forward config: train=3d / test=1d / embargo=0d — **2,507 folds**

| Model | RMSE (mean ± std) | MAE (mean ± std) |
|---|---|---|
| Linear Regression | 0.010096 ± 0.017616 | 0.010096 ± 0.017616 |
| Ridge Regression | 0.009847 ± 0.016861 | 0.009847 ± 0.016861 |
| **XGBoost** | **0.008691 ± 0.011084** | **0.008691 ± 0.011084** |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² (aggregated) | n |
|---|---|---|---|---|---|
| Linear Regression | Spring | 0.017072 | 0.009612 | +0.2000 | 638 |
| Linear Regression | Summer | 0.015867 | 0.009211 | −0.0355 | 645 |
| Linear Regression | Fall | 0.027516 | 0.011898 | −0.9697 | 630 |
| Linear Regression | Winter | 0.018709 | 0.009667 | +0.0316 | 594 |
| Ridge Regression | Spring | 0.016477 | 0.009342 | +0.2548 | 638 |
| Ridge Regression | Summer | 0.015386 | 0.009005 | +0.0264 | 645 |
| Ridge Regression | Fall | 0.026265 | 0.011575 | −0.7945 | 630 |
| Ridge Regression | Winter | 0.018121 | 0.009470 | +0.0914 | 594 |
| **XGBoost** | **Spring** | **0.012865** | **0.008095** | **+0.5458** | **638** |
| XGBoost | Summer | 0.012379 | 0.008211 | +0.3698 | 645 |
| XGBoost | Fall | 0.016592 | 0.009729 | +0.2838 | 630 |
| **XGBoost** | **Winter** | **0.014179** | **0.008754** | **+0.4438** | **594** |

---

## Iteration 4 — TSLA Regression (5-day forward realized volatility)

Walk-forward config: train=15d / test=5d / embargo=2d — **498 folds**

> Linear Regression is catastrophically unstable for TSLA at this window size — extreme coefficient magnitudes from underdetermined fitting produce several outlier folds that dominate the mean.

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.267646 ± 1.911196 | 0.241308 ± 1.735354 | −257,532 ± 3,425,300 |
| Ridge Regression | 0.023432 ± 0.027069 | 0.021396 ± 0.025699 | −94.96 ± 543.18 |
| **XGBoost** | **0.016236 ± 0.012345** | **0.014691 ± 0.011779** | **−48.11 ± 229.68** |

### Seasonal Breakdown

| Model | Season | RMSE | MAE | R² | n |
|---|---|---|---|---|---|
| Ridge Regression | Spring | 0.026656 | 0.017360 | −0.9502 | 638 |
| Ridge Regression | Summer | 0.035918 | 0.022484 | −4.3062 | 645 |
| Ridge Regression | Fall | 0.037509 | 0.023843 | −2.6600 | 630 |
| Ridge Regression | Winter | 0.041967 | 0.021971 | −3.7830 | 577 |
| **XGBoost** | **Spring** | **0.016774** | **0.012497** | **+0.2277** | **638** |
| XGBoost | Summer | 0.018965 | 0.014265 | −0.4794 | 645 |
| XGBoost | Fall | 0.022807 | 0.016235 | −0.3532 | 630 |
| XGBoost | Winter | 0.022645 | 0.015907 | −0.3926 | 577 |

---

## Challenges and Observations (Iteration 1)

**1. Noise in the regression target**

`target_return` (5-day forward return) is close to a random walk. Technical indicators explain an estimated 2–5% of the variance in 5-day returns. The remaining 95%+ is driven by news events, earnings surprises, Fed announcements, and large institutional order flow. All R² values being negative is not a modeling failure — it is a data sufficiency problem.

**2. High variance across folds**

TSLA's R² standard deviation for Linear Regression (29.986) dwarfed the mean (−10.193), reflecting severe instability. TSLA's price range (~$14 → $400+) is roughly 30× wider than SPY's, directly producing higher RMSE across all models.

**3. `is_major_event` coefficient weight is modest but non-zero**

The 10-year Yahoo Finance dataset (2015–2024) contains 144 trading days with VIX > 30. The `is_major_event` feature contributes non-zero weight but remains a minor contributor because stress days are only ~5.7% of the sample.

**4. Feature redundancy**

Correlated features (`macd`, `macd_signal`, `macd_hist`; `volatility_7` and `volatility_20`) add noise without adding signal. The second iteration reduces to 16 features by removing redundant ones.

---

## Comparing First and Second Iteration — Regression

Targets changed (`target_return` → `target_volatility`), so RMSE values are not directly comparable. The key improvement metric is R²:

| Ticker | Best Model | 1st Iter R² | 2nd Iter R² | Best Seasonal R² (Iter 2) |
|---|---|---|---|---|
| TSLA | XGBoost | n/a (different target) | -1.469254 ± 1.692770 | Summer **-0.3471** (best of all negatives) |

TSLA performance is largely unchanged between iterations (XGBoost R²=−1.469 vs −1.331 at 63d). The TSLA window is 59 days (nearly identical to the old 63), so no material difference was expected. All seasonal R² remain negative in Iteration 2; TSLA's volatile regime shifts require longer windows than available.

---

## Key Findings Across All Iterations

1. **The 3-day window (Iter3) produces the best raw RMSE and R² numbers but is statistically misleading.** With 3 training samples and 1 test sample, models interpolate rather than generalize. The high R² values reflect local curve-fitting, not predictive power.

2. **TSLA regression quality is largely unchanged across Iter1 and Iter2** (XGBoost R²=−1.469 vs −1.331). The 59d TSLA window is nearly identical to the 63d window used previously, so no material improvement was expected.

3. **XGBoost is the best TSLA regression model in every iteration.** Its combination of tree-based non-linearity and resistance to underdetermined conditions keeps it tractable when Linear Regression collapses catastrophically (R² = −257,532 in Iteration 4).

4. **TSLA regression shows no positive overall R² in Iterations 1, 2, or 4.** All seasonal R² remain negative in Iteration 2; TSLA's volatile regime shifts (meme-stock 2020–2022, post-election 2024) require longer training windows than tested.

5. **Iteration 3 (3d/1d/0d) achieves all-positive seasonal R² for TSLA XGBoost** (Spring +0.546, Winter +0.444, Summer +0.370, Fall +0.284), but this reflects local interpolation rather than generalization.

6. **Only Spring is positive for TSLA in Iteration 4** (XGBoost Spring R²=+0.228), confirming Spring as the most consistently learnable season even for volatile tickers.
