# ds340w-project

## Topic
Improving classification performance on imbalanced tabular data using synthetic data techniques.

## Description
This project builds on a parent research paper that uses SMOTE to handle class imbalance.  
We replicate the baseline approach and then introduce improvements using additional techniques.

## Structure

- `data/` → Dataset (Adult Income)
- `parent_paper_code/` → Baseline SMOTE implementation
- `modified_code/` → Improved model
- `outputs/` → Results and metrics
- `docs/` → Explanation of changes

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the project:

python main.py


## Goal
Compare baseline SMOTE vs improved approach and evaluate performance using ROC AUC and F1 score.
