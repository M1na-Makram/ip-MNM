import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def simple_rule_based_predict(row):
    """Simple heuristical rules predicting heart disease risk from raw features."""
    risk_score = 0
    if row.get('chol', 0) > 240 and row.get('age', 0) > 50: risk_score += 1
    if row.get('trestbps', 0) > 140: risk_score += 1
    if row.get('thalach', 200) < 120: risk_score += 1
    if row.get('oldpeak', 0) > 1.5: risk_score += 1
    if row.get('exang', 0) == 1: risk_score += 1
    # Returns 1 for disease (if score >= 2), else 0
    return 1 if risk_score >= 2 else 0

def main():
    os.makedirs('ml_model', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Load dataset
    data_dir = '../data/'
    report_dir = '../reports/'
    model_path = 'heart_model.joblib'
    
    df_clean = pd.read_csv(os.path.join(data_dir, 'cleaned_data.csv'))
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 1. Train Decision Tree Model ---
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    
    grid = GridSearchCV(dt, param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred_ml = best_model.predict(X_test)
    ml_acc = accuracy_score(y_test, y_pred_ml)
    
    print("--- ML Model (Decision Tree) Metrics ---")
    print(classification_report(y_test, y_pred_ml))
    joblib.dump(best_model, model_path)
    
    # --- 2. Simple Rule-Based System Evaluation ---
    # Re-load unscaled raw data explicitly aligning with cleaned rows for rules mapping
    df_raw = pd.read_csv(os.path.join(data_dir, 'raw_data.csv')).dropna(subset=['target']).reset_index(drop=True)
    raw_test = df_raw.iloc[X_test.index]
    
    y_pred_rules = raw_test.apply(simple_rule_based_predict, axis=1)
    rule_acc = accuracy_score(y_test, y_pred_rules)
    correct_rules = sum(y_test == y_pred_rules)
    
    print("\n--- Rule-Based System Metrics ---")
    print(f"Correctly classified: {correct_rules} out of {len(y_test)}")
    print(f"Rule-Based Accuracy:  {rule_acc:.2%}")
    print(f"ML Model Accuracy:    {ml_acc:.2%}")
    
    # --- 3. Save Summary Markdown ---
    total = len(y_test)
    report = f"""# Accuracy Comparison

| System | Accuracy | Correctly Classified | Total Test Samples |
|--------|----------|----------------------|-------------------|
| Machine Learning (Decision Tree) | {ml_acc:.2%} | {sum(y_test == y_pred_ml)} | {total} |
| Simple Rule-Based System | {rule_acc:.2%} | {correct_rules} | {total} |
"""
    with open(os.path.join(report_dir, 'accuracy_comparison.md'), 'w') as f:
        f.write(report)
    print(f"\nReport saved to '{os.path.join(report_dir, 'accuracy_comparison.md')}'")
        
if __name__ == "__main__":
    main()
