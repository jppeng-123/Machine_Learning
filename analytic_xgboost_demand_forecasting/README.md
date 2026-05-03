# Market-Level Demand Uplift Forecasting for Home Equity Investment

## Overview

This project builds a synthetic machine learning pipeline to identify markets with elevated qualified demand for a home equity investment product.

Instead of predicting raw website traffic or raw lead volume, the model focuses on a more business-oriented question:

> Which state-submarket-week observations are likely to show unusually strong qualified demand relative to their own recent historical baseline?

This framing is important because large, expensive, or highly populated markets naturally generate more leads. A relative demand uplift target helps separate normal market size effects from abnormal demand strength.

The project is designed as a proof-of-concept demand forecasting framework. It demonstrates how macroeconomic, housing, homeowner, geography, and funnel-related signals can be converted into an actionable market prioritization model.

## Business Problem

For a home equity investment business, demand is not only driven by raw lead activity. It also depends on market conditions, homeowner equity, affordability pressure, qualification likelihood, and channel quality.

A market-level demand forecasting model can support:

- Marketing budget allocation
- Sales and underwriting capacity planning
- Geographic prioritization
- Market expansion analysis
- Demand monitoring across state and submarket segments
- Early identification of markets where demand may be increasing

The goal is to classify elevated qualified demand, not simply high raw lead volume.

## Data

The dataset is fully synthetic and generated from economically motivated assumptions.

It simulates 10,000 observations across:

- Selected U.S. states and Washington DC
- Regional housing markets
- Eight submarket types
- Weekly time periods
- Housing, homeowner, macroeconomic, geography, and funnel-related features

Each observation represents one `state-submarket-week` market condition.

The final model uses 20 features, including:

- **Geography:** state, region, submarket type, property type
- **Macro conditions:** mortgage rate, inflation rate, consumer confidence, local unemployment rate
- **Housing market variables:** home price appreciation, housing inventory, average home value, average LTV, owner occupancy rate
- **Homeowner profile:** median income, average credit score, average DTI
- **Product and funnel signals:** property eligibility score, traditional borrowing cost pressure, website visits, ad spend

## Demand Target

Qualified demand is constructed using simplified home equity investment logic.

The code simulates a quantified demand value using:

- Lead volume
- Expected investment amount
- Homeowner cash-out need
- Channel lead quality
- Underwriting capacity fit
- Random demand shocks

Conceptually, the demand value, or the expected qualified demand value can be summarized as:

```text
expected qualified demand value
= lead volume
× expected investment amount
× cash-out need multiplier
× lead quality multiplier
× underwriting fit multiplier
× unobserved demand shock
```

This design reflects the idea that demand should not be measured only by traffic or leads. A market with many visitors may still have weak qualified demand if homeowners have low equity, poor underwriting fit, or weak product eligibility.

To avoid direct target leakage, demand-only variables used to construct the label are removed before model training.

The final binary label is defined as:

```text
label = 1 if current demand > 1.1 × trailing 8-week demand baseline
label = 0 otherwise
```

The trailing baseline is calculated separately for each `state-submarket` pair:

```text
baseline(state, submarket, week)
= average demand over the prior 8 weeks for the same state-submarket pair
```

A one-week shift is applied before calculating the rolling baseline, so the current week's demand is not included in its own benchmark.

This creates a relative uplift target:

> A market-week is labeled as high demand only if its qualified demand is meaningfully above its own recent trend.

## Missing Value Handling

The project intentionally introduces missing values to make the synthetic dataset closer to a realistic business dataset, where macro, housing, homeowner, and funnel-related variables may not always be fully available.

The missing value strategy follows a practical fill, maintain, and modify framework:

- **Categorical missing values** are labeled as `Unknown`, allowing the model to learn whether missing category information itself carries predictive signal.
- **Numeric features with low missingness below 3%** are maintained as missing values, because XGBoost can handle numeric missing values natively during tree splitting.
- **Numeric features with missingness at or above 3%** are filled using median values calculated from the training set only.
- The same training-set imputation rules are then applied to the validation and test sets.
- If a training-set `state-submarket` median is unavailable, the training-set feature-level global median is used as a fallback.

This approach avoids unnecessary over-imputation for small missing rates while still stabilizing features with more material missingness. It also preserves local market structure by using state-submarket medians before falling back to a global median, while avoiding validation or test-set information leakage.

## Modeling Approach

The project uses an XGBoost binary classifier to predict whether a market-week will show elevated qualified demand.

The modeling pipeline includes:

- One-hot encoding for categorical variables
- Median-based handling for selected numeric missing values using training-set statistics only
- XGBoost native handling of remaining numeric missing values
- Optuna hyperparameter tuning
- Time-based train-validation-test split by week
- Validation-based threshold selection using F1 score
- Final evaluation on a held-out future test set

The split structure is:

```text
Train:      weeks 9–33
Validation: weeks 34–41
Test:       weeks 42–50
```

A time-based split is used to better reflect the real forecasting problem: the model is trained on earlier market-weeks, tuned on later validation weeks, and evaluated on future held-out weeks. This avoids randomly mixing earlier and later observations from the same `state-submarket` pair across train and test sets.

## Model Output

The model estimates:

```text
P(high demand = 1 | market, macro, housing, homeowner, and funnel features)
```

The predicted probability can be used as a market prioritization score.

For example, high-scoring markets may deserve closer review for:

- Marketing spend allocation
- Underwriting resource planning
- Sales capacity planning
- Local market monitoring
- Geographic expansion or contraction decisions

The model is not intended to replace business judgment. It is designed to serve as an early-warning and prioritization signal.

## Evaluation

The model reports the following classification metrics:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Log loss
- Confusion matrix

The project also generates diagnostic plots:

- ROC curve
- Precision-recall curve
- Threshold analysis
- Feature importance chart

The classification threshold is selected on the validation set by maximizing F1 score. This is useful because the business problem involves a tradeoff between:

- **Precision:** avoiding too many false market alerts
- **Recall:** capturing as many true high-demand markets as possible

## Feature Importance

Feature importance is aggregated back to the original feature level after one-hot encoding.

For example, individual one-hot encoded state indicators are grouped back into the original `state` feature. Therefore, the importance of a categorical variable such as `state` represents the combined predictive contribution of all encoded state categories.

Strong importance from variables such as state, region, submarket type, website visits, average home value, and median income is consistent with the idea that qualified demand is highly local and market-segment dependent.

This should be interpreted as predictive association, not causality.

A high geographic feature importance does not prove that geography directly causes demand. Instead, it suggests that geographic and submarket segmentation captures persistent differences in home values, income levels, homeowner equity, affordability, market size, and funnel behavior.

## Key Interpretation

The model supports a localized demand forecasting approach.

Rather than treating demand as one national signal, the framework suggests that demand should be monitored at the market-segment level. This is especially relevant for a home equity investment product because homeowner equity, property values, homeowner profiles, and qualification likelihood can vary significantly across regions and submarkets.

A practical business interpretation is:

> Demand forecasting should be localized. State and submarket-level signals can help identify where qualified demand may be increasing relative to recent market-specific trends.

## Limitations

This project is a proof-of-concept model using synthetic data.

Main limitations include:

- No real customer, application, funding, or transaction-level data is used
- The model learns relationships embedded in the simulation assumptions
- Feature importance should not be interpreted as causal impact
- Underwriting behavior and homeowner demand are simplified
- External validation is not performed

In production, this framework should be validated using real application, marketing, property, underwriting, and funding data. A production version should also include rolling-origin backtesting, calibration checks, and ongoing model monitoring.

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost optuna
```

## How to Run

Run the Python script from top to bottom:

```bash
python demand_forecasting.py
```

The script will:

1. Generate the synthetic market-level dataset
2. Construct the qualified demand target
3. Create the rolling 8-week demand baseline
4. Generate the binary high-demand label
5. Handle missing values
6. Train and tune the XGBoost classifier
7. Select the classification threshold on validation data
8. Evaluate final performance on the held-out test set
9. Output performance metrics and diagnostic plots

## Suggested File Structure

```text
.
├── demand_forecasting.py
├── README.md
└── outputs/
    ├── model_metrics.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── feature_importance.png
```

The `outputs/` folder is optional and can be used if diagnostic plots are saved locally.

## Disclaimer

This project uses synthetic data and simplified assumptions. It is intended for modeling demonstration purposes only and should not be interpreted as a validated production model or as an official representation of any company's internal data, underwriting process, customer behavior, or business operations.
