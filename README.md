# Comparing Classifiers — Bank Term Deposit Marketing
UC Berkeley MLAI | Required Assignment 17.1: Comparing Classifiers

Practical Application III (UC Berkeley ML/AI). This project compares four
classification models on their ability to predict whether a bank client will
subscribe to a term deposit, using data from a Portuguese bank's telephone
marketing campaigns.

## Notebook
[`17.1_Comparing_Classifiers.ipynb`](17.1_Comparing_Classifiers.ipynb)

## Business Problem
Outbound calls are costly and only ~11% of contacts subscribe. The goal is to
help the bank focus its limited calling capacity on the clients most likely to
convert, rather than dialing uniformly.

## Data
UCI Bank Marketing dataset (`bank-additional-full.csv`, 41,188 rows, 20
features). Target `y` = whether the client subscribed a term deposit.
The `duration` feature is dropped from modeling because it is known only after
a call ends and leaks the outcome.

## Approach
- **Data preparation:** encoded binary features, treated `unknown` as a
  category, engineered a `pcontacted` flag from the `pdays=999` sentinel, and
  checked outliers (retained legitimate ones, e.g. high `campaign` counts).
- **Modeling:** K-Nearest Neighbors, Logistic Regression, Decision Tree, and
  SVM, built in scikit-learn `Pipeline`s with `StandardScaler` +
  `OneHotEncoder`, tuned with `GridSearchCV` (5-fold cross-validation).
- **Evaluation:** because the target is heavily imbalanced (~89% "no"),
  accuracy is misleading, so models are compared on **F1 and ROC-AUC** in
  addition to accuracy, precision, and recall.

## Key Findings
- A naive model that always predicts "no" reaches ~88.7% accuracy, so accuracy
  alone is not a meaningful measure of success.
- The strongest predictors are **prior campaign outcome** (clients with a past
  success subscribe at ~65% vs ~9% for never-contacted clients), **timing**
  (March/Sep/Oct/Dec convert at 44–50% vs 6% in May), and **macro-economic
  conditions**.
- All four model families perform similarly (F1 ~0.47, ROC-AUC ~0.80). The
  differentiator is deployability: **SVM with an RBF kernel was computationally
  infeasible** on this data (a single grid search ran 5+ hours), while a
  **LinearSVC matched it in seconds**. **Logistic Regression** is recommended
  for its speed, interpretability, and probability output.
- The best precision/recall trade-off depends on the bank's **cost per call vs.
  value of a subscription**, which is not available here. The same model can
  serve either regime by adjusting the decision threshold to
  `cost / (value + cost)`.

## Recommendations
Prioritize warm leads and prior successes, re-time campaigns away from May,
deploy a Logistic Regression propensity model to rank and work the client list
top-down, and set the calling threshold from the bank's actual cost/benefit
economics.

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook 17.1_Comparing_Classifiers.ipynb
```

## Repository Contents
- `17.1_Comparing_Classifiers.ipynb` — analysis notebook
- `data/` — dataset (`bank-additional-full.csv`) and feature descriptions
- `CRISP-DM-BANK.pdf` — reference paper on the dataset
