# Linear Regression From Scratch

A from-scratch implementation of **Linear Regression** using NumPy — no `scikit-learn` for the core fitting logic.

## What it does

Linear Regression models the relationship between a set of input features `X` and a continuous target `y` by fitting a straight line (or hyperplane, in higher dimensions):

```
y_pred = w · X + b
```

where `w` is the weight vector and `b` is the bias/intercept. The model learns `w` and `b` by minimizing the **Mean Squared Error (MSE)** between predictions and actual values:

```
MSE = (1/n) * Σ (y_pred_i - y_i)²
```

## How it's trained

The weights are learned using **gradient descent**:

1. Initialize `w` and `b` to zero (or small random values).
2. Compute predictions for the current `w`, `b`.
3. Compute the gradients of the MSE loss with respect to `w` and `b`:
   - `dw = (1/n) * Xᵀ · (y_pred - y)`
   - `db = (1/n) * Σ (y_pred - y)`
4. Update the parameters: `w -= learning_rate * dw`, `b -= learning_rate * db`.
5. Repeat for a fixed number of epochs or until the loss converges.

## Core API

```python
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

- `fit(X, y)` — runs gradient descent to learn `w` and `b` from training data.
- `predict(X)` — returns predictions for new input data using the learned parameters.

## Evaluation

Model quality is typically checked using:
- **Mean Squared Error (MSE)**
- **R² score** (coefficient of determination)

and compared against `sklearn.linear_model.LinearRegression` to confirm the from-scratch implementation converges to the same solution.

## Key concepts covered

- Gradient descent optimization
- Loss function derivation (MSE and its gradient)
- Feature scaling / normalization (important for gradient descent to converge well)
- Bias-variance tradeoff, underfitting on non-linear data

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Notes

This implementation solves for parameters iteratively via gradient descent rather than the closed-form **Normal Equation** (`w = (XᵀX)⁻¹Xᵀy`), since the goal is to build intuition for the optimization process used throughout the rest of this repo (logistic regression, SVM, etc. all reuse this same gradient descent pattern).
