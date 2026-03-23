# Changes Made

## Parent Paper
- Used SMOTE for handling class imbalance
- Used Decision Tree classifier
- Evaluated with ROC AUC

## Improvements
- Used SMOTEENN instead of only SMOTE
- Switched to XGBoost model
- Improved handling of noisy samples
- Better performance expected on minority class

## Reason
SMOTE alone can introduce noise. SMOTEENN helps clean that noise and improve generalization.
