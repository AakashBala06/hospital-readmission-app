# 🏥 Hospital Readmission Predictor

This project uses a logistic regression machine learning model to predict patient readmission risk based on electronic health record (EHR) data.

---

## ✨ Features

- Logistic Regression model with feature scaling
- Class balancing using `class_weight='balanced'`
- Performance metrics: precision, recall, F1-score, ROC AUC
- Patient-level prediction function with example output

---

## 🧰 Technologies

- Python
- pandas, scikit-learn
- Google Colab / Jupyter or any Python environment

---

## 📁 Files

- `readmission_predictor.py`: Main script (data loading, preprocessing, model training & evaluation)

---

## 📌 Use Case

This model could be used by hospitals to identify high-risk patients before discharge and reduce avoidable readmissions through targeted interventions.

---

## 📂 Dataset

The training data used for this project is from the publicly available Kaggle dataset:  
**[Hospital Readmissions | by Dan Becker](https://www.kaggle.com/datasets/dansbecker/hospital-readmissions/data)**

### 🔄 To run the model locally:
1. Download the dataset from Kaggle (you may need to log in)
2. Extract the ZIP file
3. Place the file named `train.csv` in a folder called `data_folder/`
4. Run `readmission_predictor.py` in your terminal or Jupyter Notebook

Your project structure should look like this:

hospital-readmission-predictor/ ├── readmission_predictor.py ├── data_folder/ │ └── train.csv

Results
![image](https://github.com/user-attachments/assets/badb9609-13f6-42b7-990b-a1e5e1c0b6b5)

The classification report summarizes the model's performance on a test dataset of 5,000 patient records. Here's a breakdown:

- **Precision**: Of the patients the model predicted as readmitted (`class 1`), 59% were actually readmitted.
- **Recall**: The model correctly identified 55% of all actual readmitted patients, and 69% of non-readmitted patients.
- **F1-score**: The harmonic mean of precision and recall — 0.57 for readmitted patients and 0.67 for non-readmitted.
- **Accuracy**: 62% overall accuracy across both classes.
- **ROC AUC Score**: `0.667` indicates a moderate ability to distinguish between readmitted and non-readmitted patients.

The confusion matrix shows:
- **1,872 true negatives** (not readmitted, predicted correctly)
- **1,027 true positives** (readmitted, predicted correctly)
- **859 false positives** (predicted readmitted, but weren't)
- **1,242 false negatives** (missed readmitted patients)

This output supports the use of this model for clinical triage or early-warning systems to reduce unnecessary readmissions through intervention strategies.


