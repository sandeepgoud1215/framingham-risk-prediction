# 🫀 Framingham Heart Disease Risk Prediction

This project predicts the 10-year risk of coronary heart disease (CHD) using health and demographic data from the **Framingham Heart Study**. It applies a machine learning pipeline (Random Forest) to classify patients as at risk or not at risk based on clinical and behavioral attributes.

---

## 📁 Project Structure

```
framingham-risk-prediction/
├── data/
│   └── framingham.csv              # Input dataset
├── model/
│   ├── random_forest_model.pkl     # Trained ML model
│   └── scaler.pkl                  # Scaler used for preprocessing
├── src/
│   ├── preprocessing.py            # Data loading and preprocessing
│   ├── train_model.py              # Model training and saving
│   └── predict.py                  # Prediction script using saved model
├── notebook/
│   └── analysis.ipynb              # Jupyter notebook for EDA and experimentation
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset Overview

- **Dataset**: Framingham Heart Study
- **Records**: ~4,200 patients
- **Target variable**: `TenYearCHD` (1 = developed coronary heart disease within 10 years, 0 = did not)
- **Features**:
  - Demographic: `age`, `education`, `gender`
  - Clinical: `cigsPerDay`, `BPMeds`, `prevalentStroke`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`, `diabetes`

---

## 🧠 ML Pipeline

### 1. Preprocessing
- Handle missing values using median imputation
- Feature scaling using `StandardScaler`

### 2. Model Training
- Model used: `RandomForestClassifier`
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

### 3. Model Saving
- The trained model is saved using `joblib` as `random_forest_model.pkl`
- The scaler is also saved as `scaler.pkl`

---

## 🧪 Model Performance

| Metric       | Value  |
|--------------|--------|
| Accuracy     | ~85%   |
| ROC AUC      | ~0.88  |
| Precision    | Good   |
| Recall       | Balanced |

> *Note: Scores may vary slightly depending on random seed and data split.*

---

## 🚀 How to Use

### 1. Train the Model

Run the training pipeline from terminal:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the data
- Train a `RandomForestClassifier`
- Save the model and scaler into the `model/` directory

---

### 2. Predict on New Data

You can use the saved model to predict on new user input.

```python
from src.predict import predict

# Example input: [age, education, cigsPerDay, BPMeds, prevalentStroke, ...]
sample_input = [55, 2, 10, 0, 0, 0, 240, 130, 80, 25.5, 75, 85, 0]

prediction = predict(sample_input)
print("Risk (1 = At Risk, 0 = No Risk):", prediction)
```

---

## 💻 Requirements

Install required libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## 📌 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn (for EDA)
- Joblib (for model serialization)

---

## 📂 License

This project is for educational and non-commercial purposes. Dataset is based on the publicly available **Framingham Heart Study** dataset.

---

## 🙋‍♂️ Author

**Sandeep Yelikatte**  
Data Scientist | ML Engineer  
[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/sandeepgoud1215)

---

## ⭐ Acknowledgement

Thanks to the creators of the [Framingham Heart Study](https://www.framinghamheartstudy.org/) — one of the most influential longitudinal studies in cardiovascular health.
