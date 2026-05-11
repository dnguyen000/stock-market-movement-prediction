# TSLA Classification Results

Classification target: **Next-day price direction** (UP / DOWN)

Metrics reported as **mean ± standard deviation** across all folds unless otherwise noted. Overall classification report metrics (Precision, Recall, Specificity) are computed on aggregated predictions across all folds.

> **Note on training window inconsistency:** Random Forest was run with train=189d/55 folds (matching the SPY configuration) rather than the 59d/58-fold configuration used for all other TSLA classifiers. This discrepancy is noted in each section and means RF results are not directly comparable to the other models.

---

## Logistic Regression

Walk-forward config: train=59d / test=42d / embargo=5d — **58 folds**

### Baseline Results (C=0.1, penalty=l2)

| Metric | Value |
|---|---|
| F1 (UP) | 0.430 ± 0.211 |
| Accuracy | 0.485 ± 0.077 |

### Grid Search — Top Results

Search space: C ∈ {0.001, 0.01, 0.1, 1, 10, 100} × penalty ∈ {l1, l2} — **12 combinations**

| Rank | C | Penalty | F1 (mean) | Notes |
|---|---|---|---|---|
| 1 (best F1) | 10 | l1 | 0.460 | Best F1 |
| 2 | 0.1 | l2 | 0.430 | Baseline |

### Key Observations

- Logistic Regression is the weakest TSLA classifier. F1=0.430 at baseline is substantially below SPY's 0.465 — TSLA's idiosyncratic volatility and regime breaks create non-linearities that a linear boundary cannot capture.
- L1 regularization with high C (=10) outperforms L2 at the optimum, suggesting TSLA direction benefits from sparse feature selection rather than uniform coefficient shrinkage.
- High fold-to-fold variance (std=0.211) reflects TSLA's unstable relationship between technical indicators and next-day price direction during meme-stock and post-election regimes.

---

## Random Forest

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

> **Configuration note:** This model used a 189-day training window (55 folds), matching the SPY Random Forest run rather than the standard 59-day TSLA configuration. Results are not directly comparable to the other TSLA models below.

### Baseline Results (n_estimators=100, max_depth=None)

| Metric | Value |
|---|---|
| F1 (UP) | 0.473 ± 0.135 |
| Accuracy | 0.483 ± 0.075 |

### High-Conviction Threshold (Prob > 0.55)

| Metric | Value |
|---|---|
| F1 (UP) | 0.5012 |
| Sample count | 802 |

### Grid Search — Top Results

Search space: n_estimators ∈ {100, 200} × max_depth ∈ {6, 10, None} × min_samples_leaf ∈ {1, 5, 10} × max_features ∈ {log2, sqrt}

| Rank | n_est | depth | min_leaf | features | F1 | Notes |
|---|---|---|---|---|---|---|
| 1 (best F1) | 100 | 10 | 1 | — | 0.482 | Best F1 |

### Key Observations

- Despite the longer 189-day training window, RF achieves only F1=0.473 for TSLA — well below the SPY RF baseline of 0.542. Longer history exposes the model to conflicting regimes (pre-/post-meme-stock) rather than helping it generalize.
- The high-conviction filter (Prob > 0.55) retains 802 samples and improves F1 to 0.501, crossing the 50% threshold — useful for filtering to the model's most confident directional calls.
- Grid search produces only a 0.009 improvement (0.482 vs 0.473), indicating the model has reached its ceiling given the feature set and the inherent unpredictability of TSLA's direction.

---

## Support Vector Machine (SVM)

Walk-forward config: train=59d / test=42d / embargo=5d — **58 folds**

### Baseline Results (C=10, gamma=0.1, kernel=RBF)

| Metric | Value |
|---|---|
| F1 (UP) | 0.488 ± 0.210 |
| Accuracy | 0.490 ± 0.078 |

### Grid Search — Top Results

Search space: C ∈ {0.1, 1, 10, 100, 1000} × gamma ∈ {0.001, 0.01, 0.1, 1, 10} — **25 combinations**

| Rank | C | Gamma | F1 | Accuracy | Notes |
|---|---|---|---|---|---|
| 1 (best F1) | 10 | 0.1 | 0.488 | 0.490 | Same as baseline — already at optimum |
| 2 (best Accuracy) | 0.1 | any | — | 0.514 | Precision=0.319 — aggressive UP predictor |

### Key Observations

- SVM baseline (C=10, gamma=0.1) is already at the grid optimum for F1 — no configuration improves beyond 0.488. This is unusual and indicates the RBF kernel with this hyperparameter setting sits at a local maximum for the TSLA feature space.
- The best accuracy configuration (C=0.1) achieves acc=0.514 but at precision=0.319, meaning it predicts UP on nearly all samples. This inflates accuracy by exploiting TSLA's slight UP majority without learning meaningful patterns.
- High fold-to-fold variance (std=0.210) matches Logistic Regression, confirming that TSLA's directional signal is highly non-stationary.

---

## XGBoost

Walk-forward config: train=59d / test=42d / embargo=5d — **58 folds**

Class imbalance handling: **scale_pos_weight=0.930** (DOWN/UP ratio in training set)

### Baseline Results (n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8)

| Metric | Value |
|---|---|
| F1 (UP) | 0.472 ± 0.141 |
| Accuracy | 0.484 ± 0.082 |

### Grid Search — Top Results

Search space: n_estimators ∈ {100, 200, 500} × max_depth ∈ {3, 4, 6} × learning_rate ∈ {0.01, 0.10, 0.30} × subsample ∈ {0.8, 1.0} — **81 combinations** (with scale_pos_weight=0.930)

| Rank | n_est | depth | lr | subsample | F1 | Accuracy | Notes |
|---|---|---|---|---|---|---|---|
| 1 (best F1) | 500 | 3 | 0.10 | 1.0 | 0.495 | 0.501 | Best F1 |

### Key Observations

- XGBoost's baseline F1 (0.472) is the lowest among TSLA individual models at baseline, though the gap between models is small (0.430–0.488 range).
- `scale_pos_weight=0.930` is near 1.0, reflecting TSLA's near-balanced UP/DOWN class distribution — minimal correction needed.
- Best grid configuration uses shallow trees (depth=3) with many estimators (500) — deep trees overfit within each 59-day training window. The combination of low depth and high n_estimators provides stable gradient boosting without memorizing individual fold patterns.
- XGBoost contributed to the ensemble at n_est=100, depth=3, lr=0.01, subs=1.0 (conservative to reduce overfitting in ensemble context).

---

## Voting Ensemble (Hard Vote)

Walk-forward config: train=59d / test=42d / embargo=5d — **58 folds**

Voting rule: **UP if ≥ 2 of 4 classifiers predict UP**

Component models and configurations selected for balance (not raw F1):

| Model | Configuration |
|---|---|
| SVM | C=10, gamma=0.1 |
| Logistic Regression | C=10, penalty=l1 |
| Random Forest | n_est=200, depth=4, min_leaf=10, max_features=sqrt |
| XGBoost | n_est=100, depth=3, lr=0.01, subsample=1.0 |

### Overall Results

| Metric | Value |
|---|---|
| F1 (UP) | 0.545 ± 0.153 |
| Accuracy | 0.502 ± 0.081 |
| Precision (UP) | 0.532 |
| Recall / Sensitivity (UP) | 0.653 |
| Specificity (Recall DOWN) | 0.361 |

### Seasonal Breakdown

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 613 | 0.534 |
| Summer | 645 | 0.549 |
| Fall | 630 | 0.608 |
| Winter | 548 | 0.600 |

### Key Observations

1. **The voting ensemble is the best TSLA classifier overall** (F1=0.545), outperforming all individual models at baseline and matching or exceeding their grid optima. The lift from F1~0.47–0.49 (individual) to 0.545 (ensemble) is the largest relative improvement seen across either ticker.

2. **Fall is the most predictable season for TSLA** (F1=0.608), followed closely by Winter (0.600). Spring and Summer are weaker (0.534 and 0.549). This contrasts with SPY regression (where Fall is also strong) but aligns directionally — Q4 tends to have more decisive institutional positioning and momentum-following behavior.

3. **High recall (0.653) and low specificity (0.361) indicate the ensemble is strongly UP-biased.** The majority vote skews UP because all four individual models are themselves mildly UP-biased. In practice, ensemble DOWN calls carry higher conviction than UP calls.

4. **The 59-day training window is too short for TSLA's regime complexity.** Standard deviation of 0.153 across folds is wider than SPY (0.119), reflecting the model's inability to stabilize across TSLA's 2020–2022 meme-stock surge and 2024 post-election spike.

5. **TSLA F1=0.545 vs SPY F1=0.574.** The gap is smaller than expected given TSLA's reputation for unpredictability. However, TSLA's near-50/50 class distribution (scale_pos_weight≈0.930) slightly advantages F1 metric compared to SPY's mild UP imbalance.

---

## Key Findings Across All Models

1. **The voting ensemble is the best TSLA classifier** (F1=0.545, acc=0.502), with the largest absolute improvement over individual models of any configuration tested.

2. **SVM achieves the best individual baseline F1 (0.488)** for TSLA — its baseline configuration (C=10, gamma=0.1) is already at the grid optimum, indicating the RBF kernel has found the best available non-linear boundary.

3. **All TSLA models underperform their SPY counterparts** by 0.03–0.07 F1 points. This confirms that TSLA's idiosyncratic regime breaks (earnings surprises, meme-stock dynamics, executive announcements) create directional unpredictability that no technical feature set can systematically anticipate.

4. **Grid search yields minimal gains for TSLA** (0.003–0.030 F1 improvement), compared to more meaningful gains for SPY (up to 0.078 for SVM). This suggests hyperparameter tuning is not the binding constraint — the directional signal in TSLA's technical features is fundamentally limited.

5. **Fall is the most predictable season for TSLA** (F1=0.608), consistent with the hypothesis that Q4 momentum and institutional rebalancing creates more persistent directional patterns than Q1–Q3.

6. **The 59-day training window produces higher fold-to-fold variance for TSLA than the 189-day window does for SPY.** Longer training history for TSLA would expose models to regime shifts rather than helping — TSLA's structure changes too quickly for a longer static window to provide stable signal. Walk-forward validation with adaptive or regime-aware windows would be the appropriate next step.

7. **Classification is more tractable than regression for TSLA**, consistent with SPY results. Predicting direction (F1≈0.545) is more learnable from technical features than predicting return magnitude (negative R² across all regression configurations).
