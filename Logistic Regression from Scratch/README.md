# Logistic Regression From Scratch

A from-scratch implementation of **Logistic Regression** for binary classification, built using NumPy.

## What it does

Logistic Regression predicts the probability that an input belongs to a given class by passing a linear combination of features through the **sigmoid function**:

```
z = w · X + b
p = sigmoid(z) = 1 / (1 + e^(-z))
```

The predicted class is `1` if `p ≥ 0.5`, else `0`.

## How it's trained

Parameters are learned by minimizing the **binary cross-entropy (log) loss**:

```
Loss = -(1/n) * Σ [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

using **gradient descent**:

1. Initialize `w` and `b` to zero.
2. Compute `p = sigmoid(w · X + b)` for the current parameters.
3. Compute gradients:
   - `dw = (1/n) * Xᵀ · (p - y)`
   - `db = (1/n) * Σ (p - y)`
4. Update: `w -= learning_rate * dw`, `b -= learning_rate * db`.
5. Repeat until convergence or for a fixed number of iterations.

## Core API

```python
model = LogisticRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)          # class labels
probabilities = model.predict_proba(X_test)  # raw probabilities
```

- `fit(X, y)` — learns `w` and `b` via gradient descent on the cross-entropy loss.
- `predict_proba(X)` — returns the sigmoid probability for each sample.
- `predict(X)` — thresholds `predict_proba` at 0.5 to return hard class labels.

## Evaluation

Typically evaluated with:
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion matrix**

and cross-checked against `sklearn.linear_model.LogisticRegression`.

## Key concepts covered

- The sigmoid/logit link function
- Why MSE isn't used as the loss (non-convexity) and why cross-entropy is preferred
- Decision boundaries for linearly separable data
- Gradient descent (same optimization pattern as the Linear Regression implementation in this repo)

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Notes

This is a **binary** classifier. Extending it to multi-class problems would require either a one-vs-rest wrapper or replacing the sigmoid/binary cross-entropy with softmax/categorical cross-entropy.
