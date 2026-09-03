# Decision Tree From Scratch

A from-scratch implementation of a **Decision Tree** classifier (extendable to regression) built using NumPy — no `scikit-learn` for the tree-building logic.

## What it does

A decision tree recursively splits the training data into subsets based on feature values, aiming to produce groups (leaves) that are as "pure" (single-class) as possible. Predictions are made by walking a new sample down the tree, from the root, according to the learned split conditions, until it reaches a leaf node.

## How it's built

The tree is grown recursively:

1. At each node, evaluate every feature and every possible threshold as a candidate split.
2. Score each candidate split using an **impurity measure**, typically:
   - **Gini impurity:** `Gini = 1 - Σ p_i²`
   - or **Entropy / Information Gain:** `Entropy = -Σ p_i * log2(p_i)`
3. Pick the split that most reduces impurity (maximizes information gain) between the parent node and its two children.
4. Recurse on each child node with the corresponding data subset.
5. **Stop** splitting a node (making it a leaf) when a stopping condition is met — e.g. max depth reached, minimum samples per split not met, or the node is already pure.
6. A leaf's prediction is the majority class of the samples that ended up there (or the mean, for regression trees).

## Core API

```python
model = DecisionTree(max_depth=10, min_samples_split=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

- `fit(X, y)` — recursively builds the tree by finding the best split at each node.
- `predict(X)` — traverses the tree for each sample and returns the leaf's class/value.

## Hyperparameters

- `max_depth` — limits how deep the tree can grow; prevents overfitting.
- `min_samples_split` — minimum number of samples required at a node to attempt a further split.
- `criterion` — impurity measure used to evaluate splits (`gini` or `entropy`).

## Evaluation

- **Classification:** accuracy, precision/recall, confusion matrix.
- Compared against `sklearn.tree.DecisionTreeClassifier` to validate correctness.
- Optionally: visualize the learned tree structure and decision boundaries.

## Key concepts covered

- Recursive binary splitting
- Impurity measures (Gini vs. entropy) and information gain
- Overfitting in unconstrained trees, and how `max_depth` / `min_samples_split` control it
- Why decision trees are the building block for ensemble methods (Random Forests, Gradient Boosting)

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```
