# LoanTap – Credit Risk Analysis & Loan Default Prediction

## Project Overview

LoanTap is a digital lending platform that uses technology and data analytics to simplify the loan application and approval process.

This project focuses on analysing borrower information and developing a classification model to predict whether a customer is likely to fully repay a loan or default.

Using Exploratory Data Analysis (EDA), feature engineering, statistical testing, and Logistic Regression, the project identifies factors associated with loan repayment behaviour and evaluates different approaches to handling class imbalance.

The primary objective is to help LoanTap improve its credit risk assessment process, identify potentially high-risk borrowers, and make more informed lending decisions.

## Business Objectives

- Understand borrower characteristics and loan repayment patterns.
- Identify factors associated with loan defaults.
- Analyse relationships between loan amounts, interest rates, income, and repayment behaviour.
- Perform feature engineering and statistical feature selection.
- Investigate multicollinearity among independent variables.
- Build a Logistic Regression classification model.
- Address class imbalance using SMOTE and class weighting.
- Evaluate model performance using classification metrics.
- Recommend strategies to improve credit risk assessment and loan approval decisions.

## Dataset Overview

The dataset initially contains **132,421 records and 27 columns**, representing borrower information, loan characteristics, and credit history.

The target variable is `loan_status`, which contains two categories:

| Loan Status | Description |
|---|---|
| Fully Paid | Borrower successfully repaid the loan |
| Charged Off | Loan was charged off due to non-repayment |

### Important Features

| Feature | Description |
|---|---|
| loan_amnt | Loan amount requested |
| term | Loan repayment duration |
| int_rate | Interest rate |
| installment | Monthly loan installment |
| grade | Loan grade assigned by the lender |
| sub_grade | Subcategory of loan grade |
| emp_title | Borrower's employment title |
| emp_length | Employment length |
| home_ownership | Home ownership status |
| annual_inc | Annual income |
| verification_status | Income verification status |
| issue_d | Loan issue date |
| loan_status | Loan repayment status |
| purpose | Purpose of the loan |
| dti | Debt-to-income ratio |
| open_acc | Number of open credit accounts |
| pub_rec | Number of derogatory public records |
| revol_bal | Revolving credit balance |
| revol_util | Revolving credit utilization |
| total_acc | Total number of credit accounts |
| mort_acc | Number of mortgage accounts |
| pub_rec_bankruptcies | Number of public record bankruptcies |
| address | Borrower's address |

## Tools & Technologies

- **Python**
- **Pandas & NumPy** – Data manipulation and numerical analysis
- **Matplotlib & Seaborn** – Data visualization
- **SciPy** – Statistical analysis and hypothesis testing
- **Statsmodels** – Statistical analysis
- **Scikit-learn** – Logistic Regression, preprocessing, and model evaluation
- **Imbalanced-learn** – SMOTE for handling class imbalance
- **Jupyter Notebook** – Analysis and experimentation

## Project Workflow

### 1. Data Loading & Inspection

- Imported the required Python libraries.
- Loaded the LoanTap dataset.
- Examined dataset dimensions, data types, and statistical summaries.
- Checked for missing values and duplicate records.
- Identified numerical and categorical variables.
- Analysed unique values and distributions across features.

### 2. Data Cleaning

Identified missing values in several columns:

- `emp_title`
- `emp_length`
- `title`
- `revol_util`
- `mort_acc`
- `pub_rec_bankruptcies`

Applied appropriate imputation strategies:

- Replaced missing numerical values with zero in selected columns.
- Replaced missing employment titles with `No Employee Title`.
- Replaced missing loan titles with `Unavailable`.
- Replaced missing employment length with `< 1 year`.

### 3. Feature Engineering

Created meaningful features from the existing variables:

**Credit History**
- Converted `pub_rec` into a binary indicator.
- Converted `mort_acc` into a binary indicator.
- Converted `pub_rec_bankruptcies` into a binary indicator.

**Date Features**
- Extracted issue month and issue year from `issue_d`.
- Extracted credit history month and year from `earliest_cr_line`.

**Geographical Features**
- Extracted state and ZIP code from the borrower's address.

**Employment Features**
- Extracted numerical employment length in years.
- Converted loan term into a categorical value.

### 4. Exploratory Data Analysis (EDA)

**Univariate Analysis**
- Analysed distributions of loan amounts, interest rates, annual income, and installments.
- Examined borrower employment and home ownership patterns.
- Studied loan grades, loan purposes, and repayment status.
- Identified skewness and potential outliers in numerical variables.

**Bivariate Analysis**
- Compared loan characteristics between Fully Paid and Charged Off customers.
- Examined the relationship between loan amount and installment.
- Analysed repayment behaviour across loan grades.
- Compared annual income, interest rates, and EMI across loan statuses.

**Multivariate Analysis**
- Used correlation heatmaps to identify relationships between numerical variables.
- Examined correlations among credit history variables.
- Investigated potential multicollinearity.

### 5. Statistical Analysis & Feature Selection

Applied statistical techniques to identify relevant predictors.

- Used the Chi-Square Test of Independence to examine relationships between categorical variables and loan status.
- Examined correlation between numerical variables.
- Used Variance Inflation Factor (VIF) to investigate multicollinearity.
- Removed selected features based on statistical testing, correlation, and VIF analysis.

### 6. Outlier Treatment

- Identified outliers in numerical variables.
- Applied the Z-score method with a threshold of 2.
- Implemented both outlier removal and outlier clipping approaches.
- Compared the resulting datasets before proceeding with modelling.

### 7. Data Preprocessing

- Converted target variable `loan_status` into binary values:
  - Fully Paid = 1
  - Charged Off = 0
- Encoded `initial_list_status`.
- Applied One-Hot Encoding to categorical variables.
- Used `drop_first=True` to avoid redundant dummy variables.
- Applied MinMaxScaler to numerical features.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratification to preserve the target class distribution.

### 8. Model Building

Built and evaluated multiple Logistic Regression models.

**Model 1 – Baseline Logistic Regression**
- Trained Logistic Regression on the original imbalanced dataset.
- Evaluated the model using accuracy, precision, recall, and F1-score.
- Identified poor recall for the Charged Off class.

**Model 2 – Logistic Regression with SMOTE**
- Applied Synthetic Minority Over-sampling Technique (SMOTE) to the training dataset.
- Balanced the minority and majority classes.
- Retrained Logistic Regression using the resampled training data.
- Evaluated the impact of oversampling on default detection.

**Model 3 – Regularized Logistic Regression**
- Experimented with different regularization strengths.
- Used the inverse regularization parameter `C`.
- Selected the best parameter based on test accuracy.
- Evaluated the final model using a classification report and confusion matrix.

**Model 4 – Logistic Regression with Class Weights**
- Applied `class_weight='balanced'`.
- Adjusted the model to account for the imbalance between repayment classes.
- Compared the results with the SMOTE-based model.

### 9. Model Evaluation

Evaluated model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Cross-validation

Special attention was given to precision and recall because incorrectly identifying a defaulter as a creditworthy borrower can expose a lending institution to financial risk.

## Key Findings

### 1. Target Variable Distribution

The original dataset showed an imbalanced target distribution:

| Loan Status | Percentage |
|---|---:|
| Fully Paid | 80.44% |
| Charged Off | 19.56% |

This imbalance made it important to evaluate class-specific performance rather than relying solely on overall accuracy.

### 2. Borrower & Loan Characteristics

- Mortgage holders account for approximately 50.15% of borrowers.
- Renters account for approximately 40.35%.
- Debt Consolidation and Credit Card are the two most common loan purposes.
- Individual applications are the most common application type.
- Teachers and Managers are among the most frequently observed employment titles.

### 3. Loan Repayment Behaviour

- Grade A borrowers demonstrate a high repayment rate, with approximately 93.65% of loans fully paid.
- Charged Off borrowers exhibit a higher median interest rate than Fully Paid borrowers.
- Charged Off borrowers have a lower median annual income.
- Charged Off borrowers tend to have higher median EMI and loan amounts.

### 4. Correlation Analysis

- Loan amount and installment have a strong positive correlation of approximately **0.95**.
- Total accounts and open accounts show a significant positive correlation.
- Public records and public record bankruptcies are also correlated.

These relationships highlighted the need to investigate multicollinearity before model development.

### 5. Model Performance Comparison

The following results were obtained from the notebook's test-set classification reports.

| Metric | Baseline Logistic Regression | SMOTE Logistic Regression | Class-Weighted Logistic Regression |
|---|---:|---:|---:|
| Accuracy | 89% | 79% | 79% |
| Precision – Charged Off | 97% | 46% | 80% |
| Recall – Charged Off | 44% | 80% | 80% |
| F1-score – Charged Off | 60% | 59% | 80% |
| Precision – Fully Paid | 89% | 95% | 95% |
| Recall – Fully Paid | 100% | 79% | 79% |
| F1-score – Fully Paid | 94% | 86% | 86% |

**Key observations:**

- The baseline model achieved 89% accuracy but identified only 44% of actual Charged Off loans.
- SMOTE improved Charged Off recall to 80%, but precision for that class fell to 46%.
- The class-weighted model achieved a more balanced result, with approximately 80% precision and recall for Charged Off loans.
- Overall accuracy alone does not adequately describe model performance on an imbalanced dataset.

### 6. Regularized Logistic Regression

The regularization experiment selected a lambda value of approximately **20.01**.

The notebook reported:

- Test accuracy: approximately 89%.
- Charged Off precision: 98%.
- Charged Off recall: 43%.
- Fully Paid precision: 89%.
- Fully Paid recall: 100%.

The model achieved high overall accuracy but continued to miss a substantial proportion of Charged Off borrowers.

### 7. Cross-Validation

- Applied 10-fold cross-validation.
- Reported average accuracy of approximately **89.6%** for the regularized model.

### 8. Model Evaluation Curves

- Reported ROC AUC: approximately 0.715.
- Reported Precision-Recall Average Precision: approximately 0.974.

These values should be interpreted alongside the classification report and confusion matrix, particularly because the target classes are imbalanced.

### 9. Important Predictive Features

The model coefficients highlighted several influential features:

- ZIP code
- Annual income
- Loan grade
- Interest rate
- Loan term
- Debt-to-income ratio (DTI)
- Revolving credit balance

Geographical information, particularly ZIP code, appeared to have a notable influence on model predictions.

## Business Recommendations

Based on the analysis, the following recommendations were identified:

1. **Improve Credit Risk Assessment:** Give greater attention to borrowers with lower loan grades, higher interest rates, and higher debt-to-income ratios.

2. **Prioritize Default Detection:** Evaluate models based on their ability to identify Charged Off borrowers rather than relying only on overall accuracy.

3. **Balance Precision and Recall:** Select an appropriate classification threshold based on the bank's tolerance for default risk and the cost of rejecting creditworthy customers.

4. **Review High-Risk Applications:** Consider additional verification and financial checks for applications associated with higher predicted default risk.

5. **Geographical Risk Analysis:** Investigate geographical patterns in repayment behaviour while ensuring that location-based variables are used fairly and comply with applicable lending regulations.

6. **Improve Model Performance:** Explore more advanced classification algorithms such as Random Forest and XGBoost, followed by hyperparameter tuning.

7. **Cross-Validation:** Use stratified cross-validation to maintain representative class distributions across validation folds.

8. **Continuous Model Monitoring:** Regularly monitor default rates, false positives, false negatives, and model performance to maintain reliable lending decisions.

## Key Learnings

Through this project, I strengthened my understanding of:

- Exploratory Data Analysis and credit risk analysis.
- Handling missing values and outliers.
- Feature engineering and categorical encoding.
- Chi-Square Test of Independence.
- Correlation analysis and multicollinearity.
- Variance Inflation Factor (VIF).
- Logistic Regression and regularization.
- Handling imbalanced datasets using SMOTE.
- Class weighting and its impact on classification.
- Confusion matrix, precision, recall, and F1-score.
- ROC and Precision-Recall curves.
- Cross-validation and model evaluation.
- Applying machine learning to financial risk assessment.

## Repository Structure

```text
LoanTap-Case-Study/
│
├── Business_Case_LoanTap.ipynb
├── logistic_regression.csv
└── README.md
```

*Note: Include the dataset in the repository only if its distribution is permitted.*

## Conclusion

This project demonstrates how data analysis and machine learning can support credit risk assessment and loan default prediction.

The analysis identified meaningful relationships between borrower characteristics, loan grades, interest rates, income, and repayment behaviour.

The Logistic Regression experiments also demonstrated the challenges of modelling imbalanced datasets. While the baseline model achieved high accuracy, applying SMOTE and class weighting improved the identification of Charged Off borrowers.

The project highlights the importance of selecting evaluation metrics based on business objectives and balancing the financial risks of approving likely defaulters against the opportunity cost of rejecting creditworthy customers.

---

## Author

**Umar Reyaz**

Data Analyst | Data Science & AI/ML Enthusiast

- **LinkedIn:** https://www.linkedin.com/in/umar-reyaz-55a164272
- **GitHub:** https://github.com/umar16257-design
- **Email:** umar16257@gmail.com
