# AI-Powered Quality Prediction — Automotive Manufacturing

> Random Forest classifier for real-time defect detection in automotive production lines.  
> **96.8% accuracy · 82.1% defect detection rate · Est. €2.46M annual cost saving**

---

## Project Overview

Developed as part of a supervised research project at OTH Weiden (SS2026) under  
Prof. Dr. Dr. habil. Theresa Götz, in collaboration with Umer Ahmed.

**Business Problem:** Manual quality inspection in automotive manufacturing is slow,  
inconsistent, and expensive. Defects caught late in the production cycle cost  
significantly more to fix than those detected early.

**Solution:** A machine learning pipeline that predicts defects in real time during  
production, enabling early intervention and reducing scrap rates.

---

## Results

| Metric | Value |
|---|---|
| Model Accuracy | **96.8%** |
| Defect Detection Rate (Recall) | **82.1%** |
| Features Used (MVP) | 11 of 50+ evaluated |
| Training Samples | 10,000 production cycles |
| Estimated Annual Cost Saving | **€2.46M** |

---

## Tech Stack

- **Language:** Python 3.10
- **ML:** scikit-learn (RandomForestClassifier)
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Notebook:** Jupyter

---

## Project Structure

```
├── notebooks/
│   └── model_training.ipynb    # Full pipeline: EDA → preprocessing → training → evaluation
├── src/
│   ├── data_preprocessing.py   # Feature engineering and cleaning
│   ├── model.py                # Model definition and training
│   └── evaluation.py          # Metrics, confusion matrix, feature importance
├── results/
│   └── confusion_matrix.png   # Model evaluation output
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data:** 10,000 synthetic production cycle records with 50+ sensor features  
2. **Preprocessing:** Null handling, feature scaling, RICE/Kano-based feature selection → 11 MVP features  
3. **Model:** Random Forest Classifier with cross-validation  
4. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix  
5. **Business case:** Defect cost modelling to quantify savings potential  

---

## Feature Importance (Top 5)

The model identified the following as the strongest predictors of defects:
1. Spindle temperature variance
2. Cycle time deviation
3. Tool wear index
4. Vibration amplitude
5. Hydraulic pressure delta

---

## How to Run

```bash
git clone https://github.com/pallavundhad/ai-quality-prediction-automotive
cd ai-quality-prediction-automotive
pip install -r requirements.txt
jupyter notebook notebooks/model_training.ipynb
```

---

## Note on Data

The dataset used in this project contains proprietary production data and is not  
included in this repository. A synthetic dataset with identical structure is provided  
for reproducibility.

---

*Research Project · OTH Weiden · Summer Semester 2026*  
*Supervised by Prof. Dr. Dr. habil. Theresa Götz*
