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


