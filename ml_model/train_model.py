import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


def rule_based_predict(row):
    # rough heuristic - not meant to be precise, just a baseline to compare against
    score = 0
    if row.get('chol', 0) > 240 and row.get('age', 0) > 50: score += 1
    if row.get('trestbps', 0) > 140: score += 1
    if row.get('thalach', 200) < 120: score += 1
    if row.get('oldpeak', 0) > 1.5: score += 1
    if row.get('exang', 0) == 1: score += 1
    return 1 if score >= 2 else 0


def main():
    data_dir = '../data/'
    report_dir = '../reports/'
    model_path = 'heart_model.joblib'

    os.makedirs(report_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(data_dir, 'cleaned_data.csv'))
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # grid search over depth/split params - cv=5 felt like a reasonable tradeoff
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(dt, param_grid, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred_ml = best_model.predict(X_test)
    ml_acc = accuracy_score(y_test, y_pred_ml)

    print("Decision Tree results:")
    print(classification_report(y_test, y_pred_ml))
    joblib.dump(best_model, model_path)

    # rule-based needs the raw (unscaled) values, so we reload from source
    # and align to the same test indices
    df_raw = pd.read_csv(os.path.join(data_dir, 'raw_data.csv')).dropna(subset=['target']).reset_index(drop=True)
    raw_test = df_raw.iloc[X_test.index]

    y_pred_rules = raw_test.apply(rule_based_predict, axis=1)
    rule_acc = accuracy_score(y_test, y_pred_rules)
    correct_rules = (y_test == y_pred_rules).sum()

    print(f"\nRule-based: {correct_rules}/{len(y_test)} correct ({rule_acc:.2%})")
    print(f"ML model:   {(y_test == y_pred_ml).sum()}/{len(y_test)} correct ({ml_acc:.2%})")

    total = len(y_test)
    report = f"""# Accuracy Comparison

| System | Accuracy | Correctly Classified | Total Test Samples |
|--------|----------|----------------------|-------------------|
| Machine Learning (Decision Tree) | {ml_acc:.2%} | {(y_test == y_pred_ml).sum()} | {total} |
| Simple Rule-Based System | {rule_acc:.2%} | {correct_rules} | {total} |
"""
    report_path = os.path.join(report_dir, 'accuracy_comparison.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()