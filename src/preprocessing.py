import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Fill missing values
    df.fillna(df.median(), inplace=True)

    # Separate features and target
    X = df.drop(columns=['TenYearCHD'])
    y = df['TenYearCHD']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
