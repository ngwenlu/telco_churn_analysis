# Telco Customer Churn: Segmentation & Predictive Modeling

Fully-processed customer churn for a telecommunications provider: the unsupervised segmentation, supervised prediction, and model interpretation used to develop retention strategies that can be implemented throughout. 

## TL:DR
- Categorized ~7,000 customers into three clusters by combining KMeans (with PCA) and K-Prototypes (mixed type) and cross-validating the segmentation for both approaches.
- Trained class-weighted **Logistic Regression**, **Random Forest**, and **XGBoost** churn classifiers. The linear baseline reached **0.86 test AUC** — statistically indistinguishable from the tuned gradient-boosted model, suggesting churn risk in this dataset is largely additive.
- Hyperparameter tuning for Random Forest and XGBoost churn classifiers: achieved ROC-AUC of ~0.85 on the held-out test set.
- Explained the model with SHAP and found a key interaction: MonthlyCharges only drives churn risk under short-term contracts; long contracts neutralize the effect entirely.
- Used **SHAP** on the XGBoost model as an interpretation tool to surface a key interaction: `MonthlyCharges` only drives churn under short-term contracts; long contracts neutralize the price effect entirely.
- Translated results to four specific retention plays that address the highest-risk segment (uncommitted, month-to-month, low-tenure customers).

## Problem
A telco would like to minimise its customers' churn. Treating all customers the same is wasteful — discounts are given to those that would have stuck around anyway, and high-risk customers depart before anyone notices. The questions:
1. Do there exist different segments of behavior with greater risk from churn?
2. What drives churn internally and across segments?
3. What portion of retention spend should be focused?

## Dataset
IBM Telco customer churn dataset (~7,000 rows, 21 features). Features include demographics (gender, senior citizen, partner, dependents), account information (tenure, contract type, payment method, paperless billing), services subscribed (phone, internet, online security, tech support, streaming), and charges (monthly, total). Target: binary Churn. 
The raw dataset is cleaned and encoded upstream in a separate preprocessing notebook; this notebook starts from `cleaned_churn_data.csv` and `cleaned_encoded_churn_data.csv`.

## Approach
1. **Exploratory analysis.** Class balance (~26% churn), feature distributions, base rate.
2.  **Unsupervised segmentation:**
  - **KMeans + PCA:** scree plot to select components (k=4), then biplot to interpret principal axes. Compared k=2 and k=3 cluster solutions.
  - **K-Prototypes:** handles mixed numerical/categorical features directly without one-hot expansion. Elbow method confirmed k=3 as optimal, validating the KMeans result.
3. Cluster profiling. Joint distribution of churn × cluster, plus per-cluster aggregation of numerical means and categorical modes to build readable customer archetypes.
4. **Predictive modelling:**
  - Random Forest with RandomizedSearchCV over depth, estimators, and split parameters.
  - XGBoost with staged grid search over depth/min_child_weight → gamma → subsample/colsample → regularization → learning rate, following standard practice.
5. **Interpretation.** SHAP TreeExplainer for global feature importance and individual prediction waterfalls. Drilled into the most surprising finding (MonthlyCharges as a conditional, not direct, driver) with a contract × charge-quintile churn rate table.

## Model comparison

All models trained with class weighting to address the 26% positive-class imbalance. Metrics computed on a held-out 20% test set at the default 0.5 decision threshold.

| Model                    | Test AUC | Churn precision | Churn recall | Churn F1 |
| ------------------------ | -------- | --------------- | ------------ | -------- |
| Logistic Regression      | 0.862    | 0.52            | 0.83         | 0.64     |
| Random Forest (tuned)    | 0.842    | 0.54            | 0.77         | 0.64     |
| XGBoost (tuned + SHAP)   | 0.848    | 0.51            | 0.79         | 0.62     |

**Takeaway:** A well-prepared linear baseline approaches the apparent signal ceiling of the data. The marginal lift from tree-based methods is negligible. In production we would implement logistic regression: faster, fully interpretable through coefficients, and easier to monitor for drift. XGBoost is retained to carry out SHAP analysis because TreeExplainer attributes feature interactions cleanly, but it is not considered a recommended deployment artifact. 

### Key findings
| Cluster | Profile | Churn risk |
| --- | --- | --- |
| Committed high-value | Partnered, long-tenure, high spend, 2-year contracts, auto-pay | Low |
| Conservative basic | Single, mid-tenure, low spend, 2-year contracts, mailed-check payment | Medium |
| Uncommitted | Single, low-tenure, fiber/DSL, month-to-month, paperless + electronic check | High |

The dominant churn levers (from SHAP):
- Contract length is the strongest single predictor: 2-year contracts strongly suppress churn risk regardless of other features.
- tenure decreases risk monotonically.
- OnlineSecurity and TechSupport add stickiness directly.
- MonthlyCharges is not a direct driver; it only matters when interacting with contract type. For month-to-month customers, churn rises sharply with monthly charge; for 2-year contracts, the effect disappears.

### Recommendations
1. Make month-to-month into long-term contracts, with meaningful incentives. This alone neutralizes the price-sensitivity effect.
2. Package various add-on services (security, tech support, streaming) to raise switching cost and personal investment.
3. Target the first few months of tenure (the deciding window) with onboarding and engagement programs.
4. Make auto-pay frictionless to set up; the payment-friction effect on retention is real and cheap to exploit.

### Tech stack
pandas, numpy, scikit-learn, kmodes (K-Prototypes), xgboost, shap, matplotlib, seaborn, plotnine, graphviz.
