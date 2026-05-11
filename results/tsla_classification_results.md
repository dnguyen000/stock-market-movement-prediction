# TSLA Classification Results

Classification target: **Next-day price direction (UP / DOWN)**

All models are evaluated using **walk-forward validation**:
train=59d / test=42d / embargo=5d — **58 folds**

Results are reported as **mean ± standard deviation across folds** unless otherwise noted.  
Aggregate metrics (Precision, Recall, Specificity) are computed over all out-of-fold predictions.

---

# 1. Evaluation Context

## Walk-Forward Setup
Each model is trained on a rolling 59-day window and tested on the next 42 days with a 5-day embargo to prevent leakage from overlapping features.

This configuration is intentionally short to test responsiveness to TSLA’s rapidly shifting regimes (meme-stock phase, post-2022 correction, 2024 volatility spikes).

---

## Key Metrics

| Metric | Meaning |
|---|---|
| F1 (UP) | Balance of precision and recall for UP class |
| Accuracy | Overall directional correctness |
| Precision (UP) | Reliability of UP predictions |
| Recall (UP) | Ability to detect UP days |
| Specificity | Ability to detect DOWN days |

> **Important:** TSLA is nearly balanced in class distribution, but exhibits strong regime instability. Variance across folds is as important as mean performance.

---

## Structural Challenge: TSLA Regime Instability

Unlike SPY, TSLA exhibits:

- Structural breaks (2020–2022 meme-stock regime)
- Event-driven spikes (earnings, product announcements)
- Non-stationary volatility regimes

As a result:
- Hyperparameter tuning yields limited gains
- Fold variance dominates mean performance
- Linear signal assumptions break down more frequently than for SPY

---

# 2. Logistic Regression

Walk-forward config: 59d / 42d / 5d — 58 folds

## Baseline (C=0.1, L2)

| Metric | Value |
|---|---|
| F1 | 0.430 ± 0.211 |
| Accuracy | 0.485 ± 0.077 |

## Grid Search (12 configs)

| Rank | C | Penalty | F1 |
|---|---|---|---|
| 1 | 10 | L1 | 0.460 |
| 2 | 0.1 | L2 | 0.430 |

## Key Takeaways

- Weakest TSLA classifier
- Linear decision boundary fails under regime shifts
- L1 regularization performs better than L2, suggesting sparse feature selection is more stable
- Extremely high variance (±0.211) indicates unstable signal across folds

---

# 3. Random Forest

Walk-forward config: 189d / 42d / 5d — 55 folds  
*(Note: longer training window than other TSLA models)*

## Baseline

| Metric | Value |
|---|---|
| F1 | 0.473 ± 0.135 |
| Accuracy | 0.483 ± 0.075 |

## High-Confidence Filter (Prob > 0.55)

| Metric | Value |
|---|---|
| F1 | 0.501 |
| Coverage | 802 samples |

## Grid Search (120 configs)

| Rank | Config | F1 |
|---|---|---|
| 1 | depth=10, n=100, min_leaf=1 | 0.482 |

## Key Takeaways

- Modest improvement over Logistic Regression
- Longer history does not improve performance (conflicting regimes dilute signal)
- High-confidence subset crosses 0.50 F1 threshold
- Feature interactions exist but are inconsistent across regimes

---

# 4. Support Vector Machine (RBF)

Walk-forward config: 59d / 42d / 5d — 58 folds

## Baseline (C=10, γ=0.1)

| Metric | Value |
|---|---|
| F1 | 0.488 ± 0.210 |
| Accuracy | 0.490 ± 0.078 |

## Grid Search (25 configs)

| Rank | C | γ | F1 | Accuracy |
|---|---|---|---|---|
| 1 | 10 | 0.1 | 0.488 | 0.490 |
| 2 | 0.1 | — | — | 0.514 |

## Key Takeaways

- Baseline is already optimal for F1
- No hyperparameter improves separation quality
- Lower C increases accuracy by biasing toward UP predictions (not meaningful signal improvement)
- High variance confirms unstable decision boundary across folds

---

# 5. XGBoost

Walk-forward config: 59d / 42d / 5d — 58 folds

## Baseline

| Metric | Value |
|---|---|
| F1 | 0.472 ± 0.141 |
| Accuracy | 0.484 ± 0.082 |

## Grid Search (81 configs)

| Rank | Config | F1 |
|---|---|---|
| 1 | depth=3, n=500, lr=0.1 | 0.495 |

## Key Takeaways

- Best-performing model family in stability, not raw F1
- Shallow trees (depth=3) outperform deeper ones due to overfitting risk
- Large ensembles (n=500) stabilize across short training windows
- Minimal gain from tuning confirms feature ceiling limitation

---

# 6. Voting Ensemble (Hard Vote)

Walk-forward config: 59d / 42d / 5d — 58 folds

Voting rule:
> UP if ≥ 2 of 4 models predict UP

## Component Models

| Model | Configuration |
|---|---|
| SVM | C=10, γ=0.1 |
| Logistic Regression | C=10, L1 |
| Random Forest | n=200, depth=4, min_leaf=10 |
| XGBoost | n=100, depth=3, lr=0.01 |

---

## Overall Results

| Metric | Value |
|---|---|
| F1 | 0.545 ± 0.153 |
| Accuracy | 0.502 ± 0.081 |
| Precision (UP) | 0.532 |
| Recall (UP) | 0.653 |
| Specificity | 0.361 |

---

## Seasonal Breakdown

| Season | F1 |
|---|---|
| Spring | 0.534 |
| Summer | 0.549 |
| Fall | 0.608 |
| Winter | 0.600 |

---

## Key Findings

### 1. Best Overall Model
The ensemble is the strongest TSLA classifier (F1=0.545), outperforming all individual models.

### 2. Largest Ensemble Gain in Project
Improvement from ~0.47–0.49 (individual models) → 0.545 (ensemble) is the largest relative gain across SPY and TSLA.

### 3. Strong Seasonal Signal in Q4
Fall (0.608) and Winter (0.600) show significantly stronger predictability than Spring/Summer.

This suggests:
- institutional rebalancing effects
- earnings clustering in Q4
- stronger momentum persistence

### 4. High Recall / Low Specificity Tradeoff
- Recall: 0.653 (UP-heavy bias)
- Specificity: 0.361 (weak DOWN detection)

The ensemble is more reliable for identifying upward moves than downward corrections.

### 5. Regime Instability Dominates Learning
TSLA variance (±0.153–0.211) remains high across all models, confirming:

- unstable feature-target relationships
- frequent structural breaks
- limited generalization from short windows

### 6. Feature Ceiling Effect
Grid search improvements are minimal (≤0.02–0.03 F1), indicating:
> model choice matters less than feature information content

---

# 7. Cross-Asset Insights (TSLA vs SPY)

| Property | SPY | TSLA |
|---|---|---|
| Best F1 | 0.574 | 0.545 |
| Stability | Higher | Lower |
| Regime shifts | Mild | Severe |
| Learnability | Moderate | Weak–Moderate |

## Key Insight

- SPY: signal-limited but stable
- TSLA: signal exists but unstable

This creates a paradox:
> TSLA has more volatility but less usable predictive structure

---

# Final Conclusion

TSLA next-day direction is **partially learnable but highly regime-dependent**.

Key outcomes:

- Ensemble improves robustness but not fundamental predictability
- Technical indicators provide weak but consistent signal (~54–55% F1 ceiling)
- Q4 shows strongest structural predictability
- Model performance is constrained more by **market behavior than algorithm choice**

Overall:
> TSLA is not a modeling problem — it is a regime instability problem
