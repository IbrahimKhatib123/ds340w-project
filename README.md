# ds340w-project

## Topic
Improving classification performance on imbalanced tabular data using synthetic data techniques.

## Description
This project builds on a parent research paper that uses SMOTE to handle class imbalance.  
I replicated the baseline approach and then introduce improvements using additional techniques.

## Structure

- `data/` → Dataset (Adult Income)
- `parent_paper_code/` → Baseline SMOTE implementation
- `modified_code/` → Improved model
- `outputs/` → Results and metrics
- `docs/` → Explanation of changes

## Setup Instructions

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py

## Goal
Compare baseline SMOTE vs improved approach and evaluate performance using ROC AUC and F1 score.
