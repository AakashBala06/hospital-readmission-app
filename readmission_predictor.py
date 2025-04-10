# =============================
# Hospital Readmission Predictor
# =============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# =============================
# 1. Load and Prepare Data
# =============================

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    return X, y

# =============================
# 2. Scale Features
# =============================

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# =============================
# 3. Train the Model
# =============================

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# =============================
# 4. Evaluate the Model
# =============================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("=== ROC AUC Score ===")
    print(roc_auc_score(y_test, y_proba))

# =============================
# 5. Predict New Patient
# =============================

def predict_readmission(patient_dict, model, scaler, feature_columns):
    patient_df = pd.DataFrame([patient_dict])
    patient_df = patient_df[feature_columns]  # Ensure correct column order
    patient_scaled = scaler.transform(patient_df)

    prob = model.predict_proba(patient_scaled)[0][1]
    prediction = model.predict(patient_scaled)[0]

    return {
        "prediction": int(prediction),
        "probability_readmitted": round(prob, 4)
    }

# =============================
# 6. Main Execution
# =============================

def main():
    # Load and preprocess data
    X, y = load_data("data_folder/train.csv")
    X_scaled, scaler = scale_features(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Example prediction
    example_patient = X.iloc[0].to_dict()
    result = predict_readmission(example_patient, model, scaler, X.columns)
    print("Example Prediction:", result)


if __name__ == "__main__":
    main()
