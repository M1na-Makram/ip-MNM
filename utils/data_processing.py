import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
    return df


def handle_missing_values(df, target_col='target'):
    # drop rows where we don't have a label - can't really do much with those
    df = df.dropna(subset=[target_col])

    numeric_cols = df.select_dtypes(include='number').columns
    missing = df[numeric_cols].isnull().sum()
    if missing.any():
        print("Missing values found, filling with column means:")
        print(missing[missing > 0])
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df


def scale_features(df, cols):
    # minmax works fine here since we don't have crazy outliers
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def encode_categoricals(df, cat_cols):
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def show_correlations(df, target_col='target'):
    corr = df.corr()

    print("\nCorrelation matrix:")
    print(corr.round(2))

    print(f"\nTop features correlated with '{target_col}':")
    top = corr[target_col].abs().drop(target_col).nlargest(5)
    print(top)


def main():
    input_path = '../data/raw_data.csv'
    output_path = '../data/cleaned_data.csv'

    try:
        df = load_data(input_path)
    except FileNotFoundError:
        print(f"Couldn't find {input_path}, make sure the file exists")
        return

    df = handle_missing_values(df, target_col='target')

    # these are the continuous features that need scaling
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df = scale_features(df, num_cols)

    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    df = encode_categoricals(df, cat_cols)

    show_correlations(df, target_col='target')

    df.to_csv(output_path, index=False)
    print(f"\nDone! Saved cleaned data to {output_path}")


if __name__ == "__main__":
    main()