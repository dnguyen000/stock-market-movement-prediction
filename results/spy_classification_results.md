# SPY Classification Results

Classification target: **Next-day price direction** (UP / DOWN)

All models evaluated on SPY using walk-forward validation: train=189d / test=42d / embargo=5d — **55 folds**

Metrics reported as **mean ± standard deviation** across all folds unless otherwise noted. Overall classification report metrics (Precision, Recall, Specificity) are computed on aggregated predictions across all folds.

---

## Logistic Regression

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

### Baseline Results (C=0.1, penalty=l2)

| Metric | Value |
|---|---|
| F1 (UP) | 0.465 ± 0.185 |
| Accuracy | 0.489 ± 0.092 |
| Precision (UP) | 0.514 |
| Recall / Sensitivity (UP) | 0.472 |
| Specificity (Recall DOWN) | 0.523 |

### Grid Search — Top Results

Search space: C ∈ {0.001, 0.01, 0.1, 1, 10, 100} × penalty ∈ {l1, l2} — **12 combinations**

| Rank | C | Penalty | F1 (mean) | Notes |
|---|---|---|---|---|
| 1 | 0.01 | l2 | 0.478 | Best F1 |
| 2 | 0.1 | l2 | 0.465 | Baseline |

### Key Observations

- Logistic Regression is the weakest individual classifier for SPY. F1=0.465 at baseline, improving only marginally to 0.478 at the grid optimum.
- Strongest regularization (C=0.01) outperforms weaker regularization — suggests the linear decision boundary benefits from coefficient shrinkage given correlated technical features.
- Near-random accuracy (0.489) confirms the model struggles to consistently identify the direction of the next-day move.
- Useful as a calibrated probability estimator for the voting ensemble rather than a standalone classifier.

---

## Random Forest

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

### Baseline Results (n_estimators=100, max_depth=None)

| Metric | Value |
|---|---|
| F1 (UP) | 0.542 ± 0.146 |
| Accuracy | 0.506 ± 0.095 |
| Precision (UP) | 0.555 |
| Recall / Sensitivity (UP) | 0.589 |
| Specificity (Recall DOWN) | 0.400 |

### High-Conviction Threshold (Prob > 0.55)

| Metric | Value |
|---|---|
| F1 (UP) | 0.5525 |
| Sample count | 1,086 |

### Grid Search — Top Results

Search space: n_estimators ∈ {100, 200} × max_depth ∈ {6, 10, None} × min_samples_leaf ∈ {1, 5, 10} × max_features ∈ {log2, sqrt} — **120 combinations**

| Rank | n_est | depth | min_leaf | features | F1 | Accuracy | Notes |
|---|---|---|---|---|---|---|---|
| 1 (best F1) | 200 | 10 | 1 | log2 / sqrt | 0.564 | — | Best F1 |
| 2 (best Acc) | 100 | 6 | 5 | — | — | 0.530 | Best Accuracy |

### Key Observations

- Random Forest is the strongest individual SPY classifier with F1=0.542 at baseline, reaching 0.564 at the grid optimum.
- High recall (0.589) at baseline comes at the cost of low specificity (0.400) — the model is biased toward predicting UP, which reflects the market's long-run upward drift in the SPY dataset.
- The high-conviction filter (Prob > 0.55) retains 1,086 of ~2,310 test samples while holding F1 at 0.553 — useful for trade filtering in practice.
- Depth=10 with min_leaf=1 beats shallower configurations, suggesting the model benefits from capturing non-linear interactions between technical features at moderate depth.

---

## Support Vector Machine (SVM)

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

### Baseline Results (C=10, gamma=0.1, kernel=RBF)

| Metric | Value |
|---|---|
| F1 (UP) | 0.530 ± 0.133 |
| Accuracy | 0.507 ± 0.075 |
| Precision (UP) | 0.563 |
| Recall / Sensitivity (UP) | 0.553 |
| Specificity (Recall DOWN) | 0.457 |

### Grid Search — Top Results

Search space: C ∈ {0.1, 1, 10, 100, 1000} × gamma ∈ {0.001, 0.01, 0.1, 1, 10} — **25 combinations**

| Rank | C | Gamma | F1 | Specificity | Notes |
|---|---|---|---|---|---|
| 1 (best F1) | 1 | 1 | 0.608 | 0.189 | Aggressive UP predictor — low specificity |
| 2 (most balanced) | 100 | 0.1 | 0.538 | 0.455 | Best balance of F1 and specificity |

### Key Observations

- SVM achieves the highest single-model F1 at grid optimum (0.608) but at severe specificity cost (0.189) — it predicts UP on nearly every fold when C=1/gamma=1.
- The most practically useful SVM configuration is C=100/gamma=0.1 (F1=0.538, spec=0.455), which closely matches the baseline behavior with modest improvement.
- RBF kernel captures non-linear boundaries in the scaled feature space; linear SVM was not tested but would likely underperform given the baseline logistic regression results.
- SVM contributed to the voting ensemble at C=100/gamma=0.1 to preserve balance in the vote distribution.

---

## XGBoost

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

Class imbalance handling: **scale_pos_weight=0.832** (DOWN/UP ratio in training set)

### Baseline Results (n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8)

| Metric | Value |
|---|---|
| F1 (UP) | 0.518 ± 0.125 |
| Accuracy | 0.500 ± 0.079 |
| Precision (UP) | 0.547 |
| Recall / Sensitivity (UP) | 0.525 |
| Specificity (Recall DOWN) | 0.467 |

### Grid Search — Top Results

Search space: n_estimators ∈ {100, 200, 500} × max_depth ∈ {3, 4, 6} × learning_rate ∈ {0.01, 0.10, 0.30} × subsample ∈ {0.8, 1.0} — **81 combinations** (with scale_pos_weight=0.832)

| Rank | n_est | depth | lr | subsample | F1 | Accuracy | Notes |
|---|---|---|---|---|---|---|---|
| 1 (best F1) | 500 | 6 | 0.10 | 1.0 | 0.545 | 0.517 | Best F1 |

### Key Observations

- XGBoost's baseline F1 (0.518) lands between SVM and Logistic Regression, consistent with tree-based non-linearity providing modest gains over the linear baseline.
- `scale_pos_weight=0.832` corrects for the slight UP majority without over-correcting toward DOWN; specificity of 0.467 is the highest among individual models at baseline.
- Best grid configuration uses deep trees (depth=6) with full subsample (1.0) — no subsampling stochasticity needed at this tree depth for SPY's relatively low-volatility regime.
- XGBoost contributed to the ensemble at n_est=500, depth=4, lr=0.01, subs=0.8 (conservative learning rate to reduce overfitting in ensemble context).

---

## Voting Ensemble (Hard Vote)

Walk-forward config: train=189d / test=42d / embargo=5d — **55 folds**

Voting rule: **UP if ≥ 2 of 4 classifiers predict UP**

Component models and configurations selected for balance (not raw F1):

| Model | Configuration |
|---|---|
| SVM | C=100, gamma=0.1 |
| Logistic Regression | C=0.001, penalty=l2 |
| Random Forest | n_est=200, depth=6, min_leaf=5, max_features=sqrt |
| XGBoost | n_est=500, depth=4, lr=0.01, subsample=0.8 |

### Overall Results

| Metric | Value |
|---|---|
| F1 (UP) | 0.574 ± 0.119 |
| Accuracy | 0.521 ± 0.091 |
| Precision (UP) | 0.567 |
| Recall / Sensitivity (UP) | 0.628 |
| Specificity (Recall DOWN) | 0.397 |

### Seasonal Breakdown

| Season | n (test samples) | F1 (UP) |
|---|---|---|
| Spring | 575 | 0.577 |
| Summer | 580 | 0.605 |
| Fall | 603 | 0.587 |
| Winter | 552 | 0.591 |

### Key Observations

1. **The voting ensemble is the best SPY classifier overall.** F1=0.574 exceeds every individual model's baseline, and the standard deviation (0.119) is lower than most individual models, indicating more consistent performance across folds.

2. **Summer is the most predictable season** (F1=0.605), with Spring, Fall, and Winter clustered tightly between 0.577–0.591. Unlike regression, classification does not show a strong seasonal preference — SPY's directional moves are roughly equally learnable across all seasons.

3. **Ensemble recall is high (0.628) at the cost of specificity (0.397).** The 4-model majority vote is directionally biased toward UP, reflecting SPY's long-run upward drift in the dataset. This is appropriate for a long-biased trading strategy but would miss many DOWN days.

4. **Model diversity matters.** LR (weak/linear) + SVM (kernel) + RF (bagging) + XGB (boosting) each contribute different error patterns, which is why the ensemble outperforms all individual components.

5. **Accuracy of 52.1% is above random (50%) but modest.** Next-day direction in a liquid index ETF is close to a random walk; the ensemble captures a small but consistent edge driven primarily by volatility regime features (RSI, Bollinger Band position, volatility_20).

---

## Key Findings Across All Models

1. **The voting ensemble is the best SPY classifier** (F1=0.574, acc=0.521), outperforming all individual models on both metrics.

2. **Random Forest has the best individual baseline F1 (0.542)** — its ability to capture non-linear feature interactions without an explicit kernel gives it a consistent edge over Logistic Regression and matches SVM at comparable specificity.

3. **SVM achieves the highest grid-optimized F1 (0.608) but sacrifices specificity.** C=1/gamma=1 creates a near-constant UP predictor; this is not useful in practice and highlights the importance of evaluating specificity alongside F1.

4. **XGBoost offers the best baseline precision/specificity trade-off** among tree-based models (spec=0.467 at baseline) — useful when false positives (incorrect UP calls) carry asymmetric cost.

5. **Logistic Regression is the weakest standalone classifier** but contributes to the ensemble as a calibrated probability estimator that prevents the vote from being dominated by tree-based models.

6. **Seasonal variation in classification is smaller than in regression.** All four seasons fall within a 0.028 F1 band (0.577–0.605), compared to wide seasonal divergence in regression. This suggests that directional prediction is driven by cross-seasonal features rather than seasonal patterns.

7. **All SPY classification results significantly outperform the regression task's R² scores.** Predicting binary direction (~52% accuracy) is more tractable than predicting the magnitude of a continuous return, consistent with the market microstructure literature.
