# Predicting the 2026 MLB All-Star Team
## Overview
This project builds a supervised machine learning model to predict MLB All-Star selections using historical player performance data. All-Star selection is treated as a rare-event classification problem due to the naturally imbalanced distribution between selected and non-selected players.

The goal is to identify as many true All-Stars as possible while maintaining reasonable precision, reflecting real-world costs of missed elite talent.
## Problem Context
MLB All-Star selections are influenced not only by performance, but also by perception and recognition through fan, player, and coach voting. Anticipating All-Star selections can help organizations:

* Plan for future payroll increases
* Prepare arbitration strategies
* Prioritize players for marketing and promotion

## Data
* Source: Kaggle (MLB batting statistics)
* Time Window: 2021–2024 (post-pandemic stability window)
* Observations: Player-season level data
* Target Variable:
  * AS = 1 → Selected as All-Star
  * AS = 0 → Not selected
* Class Imbalance: ~1:9 (All-Star to non–All-Star)
Only offensive statistics were used, reflecting their greater influence on All-Star voting.

## Modeling Approach
The project evaluates multiple supervised learning models using a consistent and leakage-safe pipeline:

### Models Evaluated
* Logistic Regression
* Decision Tree (pre-pruned and post-pruned)
* Random Forest
* Gaussian Naive Bayes

### Key Techniques
* Stratified train-test split
* Pipeline-based preprocessing
* GridSearchCV with stratified 5-fold cross-validation
* SMOTE oversampling (tuned sampling ratios)
* Decision threshold optimization

## Evaluation Strategy
Accuracy was intentionally avoided due to severe class imbalance.
A custom recall-weighted F-β metric was designed:
* β = 1.24 (≈ 60% recall, 40% precision)
* Reflects higher cost of missing true All-Stars (false negatives)
* Metrics computed with respect to the All-Star class

## Final Model
Logistic Regression with SMOTE (optimal ratio = 0.3)
| Metric (AS = 1) | Value |
|----------------|-------|
| Recall         | 0.81  |
| Precision      | 0.52  |
| F-β Score      | 0.65  |
| Threshold      | 0.33  |

This model achieved the best balance between recall and precision under the project’s objective.

## 2026 Predictions
The final model was applied to 2026 Steamer Projections to estimate All-Star probabilities for upcoming players. The output includes:
* Predicted probability of All-Star selection
* Binary All-Star classification using the optimized threshold

## Key Takeaways
* Class imbalance fundamentally changes how models should be evaluated
* Threshold tuning materially improves business usefulness
* Simpler, well-regularized models can outperform complex models in imbalanced settings
* Evaluation metrics must align with real-world decision costs

## Limitations & Assumptions
* Season-total statistics were used (not first-half-only data)
* Defensive metrics were excluded
* Fan voting bias could not be modeled explicitly
* Each season treated as an independent observation

## Tech Stack
* Python
* scikit-learn
* pandas, numpy
* imbalanced-learn (SMOTE)

## Contributors
Carson Pimental
