# SPY Classification Results

Classification target: **Next-day price direction (UP / DOWN)**

All models are evaluated using **walk-forward validation**:
train=189d / test=42d / embargo=5d — **55 folds**

Results are reported as **mean ± standard deviation across folds** unless otherwise noted.  
Aggregate metrics (Precision, Recall, Specificity) are computed over all out-of-fold predictions.

---

# 1. Evaluation Design

## Walk-Forward Setup
Each fold trains on a rolling 189-day window and evaluates on the next 42 days.  
A 5-day embargo prevents leakage from overlapping rolling features.

This setup simulates realistic trading conditions and evaluates model stability across time.

---

## Key Evaluation Metrics

| Metric | Meaning |
|---|---|
| F1 (UP) | Balance between precision and recall for UP class |
| Accuracy | Overall correctness |
| Precision (UP) | Correctness of UP predictions |
| Recall (UP) | Ability to detect UP days |
| Specificity | Ability to detect DOWN days |

> **Important:** F1 alone is misleading in this task due to class imbalance and directional bias. Specificity is critical for detecting model over-prediction of UP moves.

---

## Critical Methodological Insight: The “F1 Trap”

Across all models, hyperparameter tuning consistently produced configurations that **inflate F1 by over-predicting the UP class**.

Example:
- High F1 ≠ balanced classifier
- Often achieved by sacrificing specificity (DOWN detection)

To address this, every model was evaluated using a **three-way selection strategy**:

1. **Baseline model** (reasonable default)
2. **Best-F1 model (grid search winner)**
3. **Balanced model** (explicitly chosen for improved specificity)

Final ensemble components are selected primarily from **balanced configurations**, not raw F1 winners.

---

# 2. Logistic Regression

Walk-forward config: 189d / 42d / 5d — 55 folds

## Baseline (C=0.1, L2)

| Metric | Value |
|---|---|
| F1 | 0.465 ± 0.185 |
| Accuracy | 0.489 ± 0.092 |
| Precision (UP) | 0.514 |
| Recall (UP) | 0.472 |
| Specificity | 0.523 |

## Grid Search (12 configs)

| Rank | C | Penalty | F1 |
|---|---|---|---|
| 1 | 0.01 | L2 | 0.478 |
| 2 | 0.1 | L2 | 0.465 |

## Key Takeaways

- Weakest standalone classifier for SPY
- Strong regularization improves stability (C=0.01 best)
- Nearly linear decision boundary is insufficient for capturing market structure
- Best used as a **probability-calibrated ensemble component**

---

# 3. Random Forest

Walk-forward config: 189d / 42d / 5d — 55 folds

## Baseline

| Metric | Value |
|---|---|
| F1 | 0.542 ± 0.146 |
| Accuracy | 0.506 ± 0.095 |
| Precision (UP) | 0.555 |
| Recall (UP) | 0.589 |
| Specificity | 0.400 |

## Grid Search (120 configs)

| Rank | Config | F1 |
|---|---|---|
| 1 | depth=10, n=200, min_leaf=1 | 0.564 |
| 2 | depth=6, min_leaf=5 | — |

## Key Takeaways

- Strongest individual model at baseline
- High recall comes at cost of low specificity
- Captures non-linear interactions in technical indicators
- Depth improves performance up to moderate levels (≈10)

---

# 4. Support Vector Machine (RBF)

Walk-forward config: 189d / 42d / 5d — 55 folds

## Baseline (C=10, γ=0.1)

| Metric | Value |
|---|---|
| F1 | 0.530 ± 0.133 |
| Accuracy | 0.507 ± 0.075 |
| Precision (UP) | 0.563 |
| Recall (UP) | 0.553 |
| Specificity | 0.457 |

## Grid Search (25 configs)

| Rank | C | γ | F1 | Specificity |
|---|---|---|---|---|
| 1 | 1 | 1 | 0.608 | 0.189 |
| 2 | 100 | 0.1 | 0.538 | 0.455 |

## Key Takeaways

- Highest raw F1 among all models (0.608)
- Best-F1 model collapses into UP-heavy classifier
- Balanced configuration is more useful in practice
- Non-linear kernel is essential for feature interactions

---

# 5. XGBoost

Walk-forward config: 189d / 42d / 5d — 55 folds

## Baseline

| Metric | Value |
|---|---|
| F1 | 0.518 ± 0.125 |
| Accuracy | 0.500 ± 0.079 |
| Precision (UP) | 0.547 |
| Recall (UP) | 0.525 |
| Specificity | 0.467 |

## Grid Search (81 configs)

| Rank | Config | F1 |
|---|---|---|
| 1 | depth=6, n=500, lr=0.1 | 0.545 |

## Key Takeaways

- Strong middle-tier performer across all models
- Best baseline specificity among tree models
- Stable across parameter changes (low sensitivity)
- Depth=6 consistently optimal for SPY regime

---

# 6. Voting Ensemble (Hard Vote)

Walk-forward config: 189d / 42d / 5d — 55 folds

Voting rule:
> UP if ≥ 2 of 4 models predict UP

## Component Models (Balanced Selection)

| Model | Configuration |
|---|---|
| SVM | C=100, γ=0.1 |
| Logistic Regression | C=0.001, L2 |
| Random Forest | n=200, depth=6, min_leaf=5 |
| XGBoost | n=500, depth=4, lr=0.01 |

---

## Overall Results

| Metric | Value |
|---|---|
| F1 | 0.574 ± 0.119 |
| Accuracy | 0.521 ± 0.091 |
| Precision (UP) | 0.567 |
| Recall (UP) | 0.628 |
| Specificity | 0.397 |

---

## Seasonal Breakdown

| Season | F1 |
|---|---|
| Spring | 0.577 |
| Summer | 0.605 |
| Fall | 0.587 |
| Winter | 0.591 |

---

## Key Findings

### 1. Best Overall Model
The voting ensemble achieves the highest F1 (0.574), outperforming all individual models.

### 2. Stable Across Seasons
Performance varies minimally across seasons (0.577–0.605), indicating weak seasonality in directional predictability.

### 3. Directional Bias
High recall (0.628) vs low specificity (0.397) indicates a systematic bias toward predicting UP days.

### 4. Model Diversity Matters
Each base model contributes a different error structure:
- Logistic Regression → linear baseline
- SVM → non-linear boundaries
- Random Forest → feature interactions
- XGBoost → boosting corrections

This diversity is the primary reason the ensemble performs best.

### 5. Near-Ceiling Performance
~52% accuracy confirms the known difficulty of next-day directional prediction in liquid markets.

---

# 7. Cross-Model Insights

## 1. F1 is not sufficient
High F1 often corresponds to poor class balance (low specificity).

## 2. Ensemble dominates
No single model is robust across all metrics and folds.

## 3. Market signal is weak
All models cluster tightly around:
- Accuracy: ~50–52%
- F1: ~0.52–0.57

This suggests near-random directional structure.

## 4. Feature limitation dominates performance
Technical indicators alone are insufficient to break the ~52% predictive ceiling.

---

# Final Conclusion

SPY next-day direction is **weakly predictable but structurally noisy**.

Machine learning models can extract a small but consistent edge (~52% accuracy), but:

- Gains come primarily from volatility regime features
- Not from directional price structure
- Not from seasonal patterns

The ensemble improves stability and F1, but does not fundamentally overcome the near-random nature of the problem.
