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

Follow these steps to run the project on your machine.

## 1. Clone the Repository

git clone https://github.com/IbrahimKhatib123/ds340w-project.git

cd ds340w-project

## 2. Create a Virtual Environment

This avoids conflicts with your system Python.

Mac / Linux:

python3 -m venv venv

source venv/bin/activate

Windows:

python -m venv venv

venv\Scripts\activate

## important point before installing

use cd .\ds340w-project-main to enter main folder with all files

write dir in terminal to make sure you are in the right place

## 3. Install Dependencies

Use these commands to make sure packages install in the correct environment:

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt

## 4. Run the Project

python main.py

## 5. Expected Output

The script will print results for:

Baseline Model (SMOTE + Decision Tree)
Improved Model (SMOTEENN + XGBoost)

You should see:

Precision
Recall
F1 Score
ROC AUC

## 6. Troubleshooting (if needed)

If you see a NumPy or SciPy warning, run:

python -m pip install numpy==1.24.3

Then rerun:

python main.py

## Goal
Compare baseline SMOTE vs improved approach and evaluate performance using ROC AUC and F1 score.
