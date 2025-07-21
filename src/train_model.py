import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import load_and_preprocess
import os

def train_and_save_model():
    X, y, scaler = load_and_preprocess('data/framingham.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, 'model/random_forest_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

if __name__ == "__main__":
    train_and_save_model()
