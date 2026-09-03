# Applied Machine Learning

A collection of classical machine learning algorithms implemented **from scratch** in Python (NumPy/Pandas only, no `scikit-learn` for the core math), alongside a large set of **practical, library-based notebooks** covering the full applied ML pipeline — from raw data cleaning through supervised, unsupervised, and ensemble learning.

## Why this repo exists

It's split into two complementary halves:

- **"From Scratch" folders** — re-derive and re-implement core algorithms manually (gradient descent, splitting criteria, distance metrics, margin optimization) to build real intuition for what a library call like `.fit()` is actually doing under the hood.
- **`Practical`** — the flip side: using `scikit-learn` and friends the way they're used in real projects, covering the entire pipeline a working ML practitioner needs — preprocessing, encoding, scaling, model selection, evaluation, tuning, clustering, association rules, and ensembling.

## Repository structure

| Folder | Focus | Type |
|---|---|---|
| [`Linear Regression from Scratch`](./Linear%20Regression%20from%20Scratch) | Linear Regression, implemented manually | From scratch |
| [`Logistic Regression from Scratch`](./Logistic%20Regression%20from%20Scratch) | Logistic Regression, implemented manually | From scratch |
| [`K Nearest Neighbors From Scratch`](./K%20Nearest%20Neighbors%20From%20Scratch) | K-Nearest Neighbors, implemented manually | From scratch |
| [`Decision Tree From Scratch`](./Decision%20Tree%20From%20Scratch) | Decision Tree, implemented manually | From scratch |
| [`Support Vector Machine From Scratch`](./Support%20Vector%20Machine%20From%20Scratch) | Support Vector Machine, implemented manually | From scratch |
| [`Practical`](./Practical) | 50+ notebooks: preprocessing, regression, classification, clustering, association rules, ensembles — using standard libraries on 14 real datasets | Applied / library-based |

Each folder contains its own `README.md` with details specific to that topic — start there.

## What's covered, end to end

**Data preprocessing:** missing values, imputation, duplicates, dtype handling, encoding (label / one-hot / dummy), outlier detection (box plot, dist plot, Z-score), scaling (standardization, min-max), function transforms, train-test split, feature elimination.

**Regression:** linear, multilinear, polynomial regression, cost functions, Lasso/Ridge regularization, KNN regression, SVM regression, decision tree regression.

**Classification:** logistic regression (binary, multi-input, polynomial), one-vs-rest multi-class, Naive Bayes, decision trees (entropy, information gain, Gini), KNN, SVM.

**Model evaluation & tuning:** confusion matrix (accuracy, precision, recall), imbalanced datasets, overfitting, hyperparameter tuning, cross-validation (K-Fold, LPO, LOOCV).

**Unsupervised learning:** K-Means, hierarchical clustering, DBSCAN, silhouette score.

**Association rule learning:** Apriori, FP-Growth.

**Ensemble learning:** voting classifiers/regressors, bagging, Random Forest (classification & regression).

## Getting started

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/Dhruv-kundu/Applied-Machine-Learning.git
cd Applied-Machine-Learning
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend jupyter
```

Then open any folder and run its notebook(s)/script(s) — see that folder's README for specifics and, for `Practical`, a full notebook-by-notebook and dataset-by-dataset breakdown.

## Author

**Dhruv Kundu** — B.Tech AI & ML student, GGSIPU. Built as part of ongoing coursework and self-study in applied machine learning.

