import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


def run_baseline(data_path):
    df = pd.read_csv(data_path)

    # Drop missing values
    df = df.dropna()

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE (parent paper method)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = DecisionTreeClassifier()
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    print("=== Baseline Model (SMOTE + Decision Tree) ===")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("ROC AUC:", roc)
