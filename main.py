from parent_paper_code.baseline_model import run_baseline
from modified_code.improved_model import run_improved

DATA_PATH = "data/adult.csv"

if __name__ == "__main__":
    run_baseline(DATA_PATH)
    print("\n" + "="*50 + "\n")
    run_improved(DATA_PATH)
