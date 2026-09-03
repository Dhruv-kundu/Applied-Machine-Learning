# Practical

A collection of hands-on Jupyter notebooks covering the full applied machine learning pipeline — from raw data cleaning through supervised, unsupervised, and ensemble learning — each focused on a single concept and run against real or classic datasets.

Unlike the "From Scratch" folders in this repo (which reimplement algorithms manually with NumPy), the notebooks here use standard libraries (`pandas`, `scikit-learn`, `mlxtend`, etc.) to practice the *end-to-end workflow*: preprocessing, model building, tuning, and evaluation.

## Contents

### 1. Data Cleaning & Preprocessing
| Notebook | Covers |
|---|---|
| `Missing Values.ipynb` | Detecting missing values in a dataset |
| `Handling Missing Values.ipynb` | Strategies for dealing with missing values — dropping rows/columns |
| `Imputing Alpha Data.ipynb` | Imputing missing categorical (alphabetic) data |
| `Numerical Data.ipynb` | Handling and imputing missing numerical data |
| `Handling Duplicates.ipynb` | Detecting and removing duplicate records |
| `Replacing Data Type.ipynb` | Changing/casting column data types |

### 2. Encoding Categorical Data
| Notebook | Covers |
|---|---|
| `Label Encoding.ipynb` | Converting categories to integer labels |
| `Dummy Variable.ipynb` | Creating dummy variables from categorical columns |
| `One-Hot Encoding.ipynb` | One-hot encoding categorical features |

### 3. Outlier Detection
| Notebook | Covers |
|---|---|
| `Outlier Detection - Box Plot & Dist Plot.ipynb` | Visual outlier detection using box plots and distribution plots |
| `Outlier Detection - Z-Score.ipynb` | Statistical outlier detection using the Z-score method |

### 4. Feature Scaling & Transformation
| Notebook | Covers |
|---|---|
| `Standardization - Feature Scaling.ipynb` | Standardizing features (zero mean, unit variance) |
| `Min-Max Scaling.ipynb` | Scaling features to a fixed [0, 1] range |
| `Function Transform.ipynb` | Applying function transformations (log, sqrt, etc.) to features |

### 5. Data Splitting & Feature Selection
| Notebook | Covers |
|---|---|
| `Train-Test Split.ipynb` | Splitting data into training and test sets |
| `Backward and Forward Elimination.ipynb` | Stepwise feature selection techniques |

### 6. Regression
| Notebook | Covers |
|---|---|
| `Linear Regression.ipynb` | Simple linear regression |
| `Multilinear Regression.ipynb` | Multiple linear regression |
| `Polynomial Regression.ipynb` | Fitting non-linear relationships with polynomial features |
| `Cost Function.ipynb` | Cost/loss functions used in regression |
| `Lasso and Ridge - Regularization Technique.ipynb` | L1 (Lasso) and L2 (Ridge) regularization |

### 7. Classification
| Notebook | Covers |
|---|---|
| `Classification Algorithms.ipynb` | Overview of classification approaches |
| `Logistic Regression.ipynb` | Binary logistic regression |
| `Multi-Input Logistic Regression.ipynb` | Logistic regression with multiple input features |
| `Polynomial Classifier - Logistic Regression.ipynb` | Logistic regression with polynomial decision boundaries |
| `Polynomial Features - PolynomialFeatures.ipynb` | Generating polynomial/interaction features with `sklearn.preprocessing.PolynomialFeatures` |
| `Multi-Class Binary Classifier - One-vs-Rest (OVR).ipynb` | Extending binary classifiers to multi-class problems via OvR |
| `Naive Bayes.ipynb` | Naive Bayes classification |
| `Decision Tree - Entropy, Information Gain, Gini Index.ipynb` | Decision tree classification and splitting criteria |
| `Decision Tree Regression.ipynb` | Using decision trees for regression tasks |
| `K-Nearest Neighbors (KNN) Algorithm.ipynb` | KNN classification |
| `KNN Regression.ipynb` | KNN for regression tasks |
| `Support Vector Machine (SVM).ipynb` | SVM classification |
| `SVM Regression.ipynb` | Support Vector Regression (SVR) |

### 8. Model Evaluation & Tuning
| Notebook | Covers |
|---|---|
| `Confusion Matrix - Accuracy, Error, Precision, Recall.ipynb` | Classification evaluation metrics |
| `Imbalanced Dataset.ipynb` | Techniques for handling class imbalance |
| `Overfitting.ipynb` | Diagnosing and addressing overfitting |
| `Hyperparameters.ipynb` | Hyperparameter tuning |
| `Cross-Validation - K-Fold, LPO, LOOCV.ipynb` | K-Fold, Leave-P-Out, and Leave-One-Out cross-validation |

### 9. Unsupervised Learning
| Notebook | Covers |
|---|---|
| `Unsupervised Learning.ipynb` | Overview of unsupervised learning |
| `K-Means Clustering.ipynb` | K-Means clustering |
| `Hierarchical Clustering.ipynb` | Agglomerative/hierarchical clustering |
| `DBSCAN Clustering.ipynb` | Density-based clustering with DBSCAN |
| `Silhouette Score.ipynb` | Evaluating cluster quality with the silhouette score |

### 10. Association Rule Learning
| Notebook | Covers |
|---|---|
| `Association Rule Learning.ipynb` | Overview of association rule mining |
| `Apriori Algorithm.ipynb` | Mining frequent itemsets and rules with Apriori |
| `FP-Growth.ipynb` | Frequent pattern mining with FP-Growth |

### 11. Ensemble Learning
| Notebook | Covers |
|---|---|
| `Ensemble Learning.ipynb` | Overview of ensemble methods |
| `Voting Classifier.ipynb` | Combining classifiers via majority/soft voting |
| `Voting Regression.ipynb` | Combining regressors via averaging |
| `Bagging Meta-Estimator - Random Forest Classifier.ipynb` | Bagging and Random Forest for classification |
| `Bagging Meta-Estimator - Random Forest Regression.ipynb` | Bagging and Random Forest for regression |

## Datasets

The `datasets/` (or similarly named) folder contains the data used across the notebooks above:

| Dataset | Typical use |
|---|---|
| `Data` | General-purpose preprocessing practice |
| `Diabetes` | Classification/regression on medical data |
| `Groceries` | Association rule mining (Apriori / FP-Growth) |
| `Groceries_House_Price` | Regression practice |
| `Ice Cream Selling Data` | Simple linear regression |
| `Iris` | Classic classification dataset (Decision Tree, KNN, SVM, clustering) |
| `Iris Row` | Raw/unprocessed variant of the Iris dataset for preprocessing practice |
| `Loan` | Classification (loan approval prediction) |
| `Manufacturing` | Regression/classification on manufacturing data |
| `Multilinear Regression Data` | Multiple linear regression practice |
| `Placement` | Classification (student placement prediction) |
| `Placement 2` | Additional placement classification variant |
| `Salary Data` | Simple/multilinear regression practice |
| `Social Media Networks` | Classification (e.g. ad click / purchase prediction) |

## How to use this folder

Each notebook is self-contained — it loads one dataset from above, walks through the concept named in the title, and prints/plots the result. To run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
jupyter notebook
```

Open any notebook and run all cells top to bottom. Notebooks are grouped by topic in the tables above; working through them roughly in that order follows the natural order of an ML pipeline: **clean → encode → scale → split → model → evaluate → tune → (unsupervised / ensemble extensions)**.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
mlxtend        # for Apriori / FP-Growth
scipy
```
