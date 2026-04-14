import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Loads the dataset from the specified path."""
    return pd.read_csv(filepath)

def handle_missing_values(df, target_col='target'):
    """Fills numeric missing values with mean and drops missing targets."""
    df = df.dropna(subset=[target_col])
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def scale_numeric_features(df, cols_to_scale):
    """Applies MinMaxScaler to specified numeric columns."""
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

def encode_categorical_features(df, categorical_cols):
    """One-hot encodes specified categorical columns."""
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def analyze_correlations(df, target_col='target'):
    """Prints correlation matrix and top 5 features correlated with target."""
    corr_matrix = df.corr()
    print("--- Correlation Matrix ---")
    print(corr_matrix.round(2))
    
    print(f"\n--- Top 5 Features Correlated with '{target_col}' ---")
    # Get absolute correlations, drop the target itself, and get top 5
    top_5 = corr_matrix[target_col].abs().drop(target_col).nlargest(5)
    print(top_5)

def main():
    input_path = '../data/raw_data.csv'
    output_path = '../data/cleaned_data.csv'
    
    try:
        df = load_data(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please ensure the data exists.")
        return

    # Handle missing values
    df = handle_missing_values(df, 'target')
    
    # Scale numeric features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df = scale_numeric_features(df, numeric_features)
    
    # One-hot encode categorical features
    categorical_features = ['cp', 'restecg', 'slope', 'thal']
    df = encode_categorical_features(df, categorical_features)
    
    # Analyze and print correlations
    analyze_correlations(df, 'target')
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\nData preprocessing complete! Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()
