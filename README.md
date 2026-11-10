This is a project in which I compare different supervised models as part of my Machine Learning with Python class. In this project, I test the performance of Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.

# Comparing Supervised Models for Loan Prediction

## What's the approach?

The goal is to predict loan repayment performance from borrower and loan characteristics, then compare how well different supervised classification models distinguish between repayment outcomes.

1. **Explore the data:** Inspect loan statuses, missing values, borrower records, and categorical feature distributions.
2. **Define the target:** Create a binary variable, `loan_performance`. Fully paid loans are intended to receive **1**, while charged-off and defaulted loans receive **0**. Exclude Current, Issued, late, and grace-period loans because their final outcomes are unresolved.
3. **Prepare the features:**
   - Remove columns with approximately 80% or more missing values.
   - Remove identifiers, free-text fields, redundant variables, and payment or recovery information that could reveal the outcome.
   - Calculate `credit_hist_months` from the loan issue date and earliest credit line.
   - Group less frequent loan purposes and home ownership categories.
   - Map employment length and loan grade to ordered numeric values.
   - Fill missing employment length with 0 and missing months since last delinquency with twice the observed maximum; remove rows with remaining missing values.
   - One-hot encode categorical variables and standardize numeric features.
4. **Train and tune:** Split the data into 70% training and 30% testing with `random_state=42`. Use `GridSearchCV` with 10-fold cross-validation to tune each model. Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting use ROC AUC for tuning; XGBoost uses its default accuracy score. The first three models use balanced class weights.
5. **Evaluate:** Compare test-set accuracy, ROC AUC, precision, recall, F1 scores, confusion matrices, ROC curves, and precision–recall curves.

### Results recorded in the notebook

| Model | Test ROC AUC | Test accuracy |
| --- | ---: | ---: |
| Logistic Regression | 0.7012 | 65.56% |
| Decision Tree | 0.6888 | 60.22% |
| Random Forest | 0.7124 | 66.15% |
| Gradient Boosting | 0.7226 | 81.75% |
| XGBoost | 0.7234 | 81.70% |

XGBoost achieves the highest ROC AUC, while Gradient Boosting achieves the highest accuracy. However, both boosting models have low recall for class 0 (approximately 9%). The notebook selects **Random Forest** because it has the highest minimum F1 score across the two classes, offering a more balanced comparison of repayment outcomes.

The selected Random Forest parameters are `n_estimators=200`, `max_depth=10`, and `max_features="sqrt"`.

## Data Details

The project uses Lending Club loan records stored in `loans.csv`, with variable definitions provided in `LoanDataDictionary.xlsx`.

| Item | Details |
| --- | --- |
| Original dataset | 887,379 loans and 74 data columns, after removing the exported index column |
| After loan-status filtering | 256,939 loans |
| Final modeling dataset | 256,699 loans |
| Predictors before one-hot encoding | 22, including the engineered credit-history feature |
| Model input after encoding | 32 features |
| Target | `loan_performance` |
| Test set | 77,010 loans |

The predictors cover loan amount, term, interest rate, installment, grade, employment length, home ownership, annual income, verification status, purpose, debt-to-income ratio, delinquency history, credit inquiries, account counts, public records, revolving balance and utilization, listing status, and credit-history length.

The test set is imbalanced: the saved results contain 62,329 class-1 observations and 14,681 class-0 observations. This is why the analysis considers class-specific recall and F1 scores alongside accuracy.

### Project files

- `Comparing_Supervised_Models_Loan_Prediction.ipynb` — Data exploration, preprocessing, model tuning, visualizations, and evaluation.
- `loans.csv` — Input loan dataset, approximately 486 MB.
- `LoanDataDictionary.xlsx` — Reference definitions for dataset fields.
- `README.md` — Project overview and setup instructions.

### Notes on the current implementation

These results are taken from the notebook's saved outputs; they have not been independently reproduced.

- The label-mapping code contains a trailing space in `'Does not meet the credit policy. Status:Fully Paid '`. The CSV category has no trailing space, so those loans are currently assigned 0 instead of the intended 1. Correcting this requires rerunning the analysis and updating the reported results.
- Scaling and some data-dependent preprocessing occur before the train/test split. For a stricter evaluation, fit preprocessing on the training data within a pipeline and cross-validation.
- The missing-delinquency fill value is a modeling assumption, rather than confirmation that a borrower has no delinquency history.

## How to run the code

### 1. Prepare the project files

Download or clone the project, and place `loans.csv` in the same folder as the notebook. The notebook loads it using the relative path `pd.read_csv("loans.csv")`.

```text
Comparing_Supervised_Models_Loan_Prediction/
├── Comparing_Supervised_Models_Loan_Prediction.ipynb
├── LoanDataDictionary.xlsx
├── loans.csv
└── README.md
```

If the CSV is distributed separately, obtain the dataset used for this project and place it in this folder. A dataset download URL is not included in the current project files.

### 2. Create a Python environment

The saved notebook metadata identifies Python 3.10.19. Python 3.10 is a suitable starting point; the project does not currently include a pinned dependency file.

Open a terminal in the project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel pandas numpy scikit-learn xgboost matplotlib seaborn
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

### 3. Open and run the notebook

```bash
python -m jupyter lab
```

Open `Comparing_Supervised_Models_Loan_Prediction.ipynb`, select the Python kernel from the environment above, and run the cells from top to bottom. The notebook also includes an XGBoost installation cell; the setup command above installs it in advance.

The notebook displays preprocessing checks, model tuning results, classification reports, and evaluation plots inline.

**Runtime:** The CSV is large, and the notebook runs multiple parameter grids with 10-fold cross-validation. A full run can take substantial time and memory, especially for Random Forest and Gradient Boosting. For an initial trial, reduce the parameter grids or use a smaller sample, keeping in mind that the results will change.
