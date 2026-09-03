# K-Nearest Neighbors (KNN) From Scratch

A from-scratch implementation of the **K-Nearest Neighbors** algorithm using NumPy — a simple, non-parametric method for classification (and regression).

## What it does

KNN makes no assumptions about the underlying data distribution and doesn't "learn" parameters in the traditional sense. Instead, for a new data point it:

1. Computes the distance from that point to every point in the training set (typically **Euclidean distance**):
   ```
   distance(a, b) = sqrt( Σ (a_i - b_i)² )
   ```
2. Selects the `k` closest training points ("neighbors").
3. **Classification:** predicts the majority class among those `k` neighbors.
   **Regression:** predicts the average (or weighted average) of their target values.

Because it defers all computation to prediction time, KNN is often called a **lazy learner** — there's no explicit training phase beyond storing the data.

## Core API

```python
model = KNN(k=5)
model.fit(X_train, y_train)          # simply stores the training data
predictions = model.predict(X_test)
```

- `fit(X, y)` — stores the training data (no optimization happens here).
- `predict(X)` — for each sample, computes distances to all training points, finds the `k` nearest, and returns the majority-vote class (or mean, for regression).

## Choosing `k`

- **Small `k`** (e.g. 1–3) → sensitive to noise, can overfit, jagged decision boundary.
- **Large `k`** → smoother decision boundary, but risks underfitting and washing out local structure.
- `k` is usually chosen via cross-validation, and an **odd `k`** is preferred for binary classification to avoid tie votes.

## Evaluation

- **Classification:** accuracy, precision/recall, confusion matrix.
- **Regression:** MSE, R².
- Cross-checked against `sklearn.neighbors.KNeighborsClassifier` / `KNeighborsRegressor`.

## Key concepts covered

- Distance metrics (Euclidean, and how the implementation could be extended to Manhattan/Minkowski)
- The importance of **feature scaling** — KNN is distance-based, so unscaled features with large ranges dominate the distance calculation
- The curse of dimensionality — why KNN degrades in high-dimensional feature spaces
- Computational cost — naive KNN is O(n) per prediction since it scans the full training set

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```
