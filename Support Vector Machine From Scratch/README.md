# Support Vector Machine (SVM) From Scratch

A from-scratch implementation of a **linear Support Vector Machine** classifier using NumPy, trained via gradient descent on the hinge loss.

## What it does

SVM finds the hyperplane that separates two classes with the **maximum margin** — the widest possible gap between the decision boundary and the nearest points of each class (the "support vectors").

The decision function is:

```
f(x) = w · x - b
```

Predicted class: `+1` if `f(x) ≥ 0`, else `-1` (labels are typically encoded as `{-1, +1}` rather than `{0, 1}` for this formulation).

## How it's trained

The model minimizes the **hinge loss** with L2 regularization:

```
Loss = λ‖w‖² + (1/n) * Σ max(0, 1 - y_i * (w · x_i - b))
```

- The `λ‖w‖²` term maximizes the margin (regularization).
- The hinge loss term penalizes points that are misclassified or fall inside the margin.

Trained via **gradient descent**:

1. For each training sample, check if it's correctly classified with margin ≥ 1:
   - If yes: `w -= learning_rate * (2 * λ * w)` (only regularization term applies)
   - If no: `w -= learning_rate * (2 * λ * w - y_i * x_i)`, `b -= learning_rate * y_i`
2. Repeat over all samples for a fixed number of epochs.

## Core API

```python
model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

- `fit(X, y)` — learns `w` and `b` by minimizing the regularized hinge loss.
- `predict(X)` — returns the sign of `w · x - b` as the predicted class.

## Hyperparameters

- `learning_rate` — step size for gradient descent.
- `lambda_param` — regularization strength; controls the margin/misclassification tradeoff.
- `n_iters` — number of passes over the training data.

## Evaluation

- Accuracy, precision/recall, confusion matrix on held-out data.
- Decision boundary visualization (for 2D toy datasets) showing the separating hyperplane and margin.
- Compared against `sklearn.svm.SVC(kernel='linear')`.

## Key concepts covered

- Margin maximization and support vectors
- Hinge loss and its subgradient
- Role of the regularization parameter `λ` (bias-variance tradeoff, soft-margin behavior)
- Why this implementation is linear only — kernel tricks (RBF, polynomial) for non-linear boundaries are a natural extension but aren't implemented here

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```
