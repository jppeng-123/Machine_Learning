# Demand Forecasting - A Machine Learning Approach

## Overview

This project builds a synthetic machine learning pipeline for identifying markets with elevated qualified demand for a home equity investment product. The model predicts whether a state-submarket-week observation has demand that is meaningfully above its own recent historical baseline.

The goal is not to predict raw website traffic or raw lead volume. Instead, the model answers a more business-focused question:

> Which markets are likely to show unusually strong qualified demand relative to their own recent trend?

This framing is useful because large or expensive markets naturally generate more leads. A relative uplift target helps separate normal market size from abnormal demand strength.

## Business Problem

For a home equity investment business, identifying high-demand markets can support:

- Marketing budget allocation
- Sales and underwriting capacity planning
- Geographic prioritization
- Market expansion analysis
- Demand monitoring across state and submarket segments

The model is designed to classify elevated qualified demand, not just high raw lead activity.

## Data

The dataset is fully synthetic and generated from economic assumptions. It simulates 10,000 observations across:

- U.S. states and Washington DC
- Regional housing markets
- Eight submarket types
- Weekly time periods
- Housing, borrower, macroeconomic, and funnel-related features

The final model uses 20 features, including:

- Geography: state, region, submarket type, property type
- Macro conditions: mortgage rate, inflation, consumer confidence, unemployment
- Housing market variables: home value, HPA, inventory, LTV
- Borrower profile: income, credit score, DTI
- Product/funnel signals: eligibility score, website visits, ad spend

## Demand Target

Qualified demand is constructed using simplified home equity investment logic. The code estimates expected qualified demand value from:

- Lead volume
- Expected investment amount
- Homeowner cashout need
- Channel lead quality
- Underwriting capacity fit
- Random demand shocks

To avoid direct target leakage, the demand-only variables used to construct the label are dropped before model training.

The final label is defined as:

```text
label = 1 if current demand > 1.1 × trailing 8-week demand baseline
label = 0 otherwise
```

The rolling baseline is calculated separately for each state-submarket pair and uses a one-week shift, so the current week is not included in its own baseline.

## Modeling Approach

The project uses an XGBoost binary classifier with:

- One-hot encoding for categorical variables
- XGBoost native handling of remaining numeric missing values
- Optuna hyperparameter tuning
- Random stratified train/validation/test split
- Validation-based threshold selection using F1 score
- Final evaluation on a held-out test set

The split structure is:

```text
Train:      60%
Validation: 20%
Test:       20%
```

A random split is used because this project is framed as a synthetic cross-sectional classification demonstration, not a strict forward-time production forecasting system.

## Evaluation

The model reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Log loss
- Confusion matrix
- ROC curve
- Precision-recall curve
- Threshold analysis
- Feature importance

Feature importance is aggregated back to the original feature level after one-hot encoding.

## Interpretation

The model is expected to learn patterns from geography, submarket structure, housing value, funnel activity, and borrower affordability. Strong importance from variables such as state, region, submarket type, website visits, average home value, and median income is consistent with the idea that home equity demand is highly local and market-segment dependent.

## Limitations

This is a proof-of-concept model using synthetic data.

Main limitations:

- No real Unison customer, application, or funding data is used
- Random split does not test strict forward-time forecasting performance
- Feature importance is not causal
- Underwriting and demand behavior are simplified
- No external validation is performed

In production, this model should be validated using real application, marketing, property, and funding data with time-based or rolling-origin out-of-sample testing.

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost optuna
```

## How to Run

Run the notebook or Python script from top to bottom:

```bash
python unison_synthetic_demand_model.py
```

The script will generate the synthetic dataset, construct the demand label, handle missing values, train the XGBoost model, and output performance metrics and diagnostic plots.

## Disclaimer

This project uses synthetic data and simplified assumptions. It is intended for modeling demonstration purposes only and should not be interpreted as a validated production model or as an official representation of Unison's internal data, underwriting process, or business operations.
