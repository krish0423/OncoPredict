# OncoPredict: XGBoost Clinical Diagnostic System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-green?style=for-the-badge)

**OncoPredict** is an enterprise-grade medical AI application capable of diagnosing breast cancer malignancy with **98.2% accuracy**. It integrates a Gradient Boosting classifier with **SHAP (Shapley Additive exPlanations)** to provide transparent, feature-level reasoning for every diagnosis, bridging the gap between "Black Box" AI and clinical trust.

---

## 🚀 Key Technical Features

### 1. High-Performance Inference Engine

- **Model:** XGBoost Classifier trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
- **Performance:** Optimized for high Recall (Sensitivity) to minimize false negatives in medical contexts.
- **Preprocessing:** Robust scaling pipelines to handle outlier variations in cellular data.

### 2. Explainable AI (XAI) Architecture

- **Transparency:** Integrates **SHAP** to calculate the marginal contribution of all 30 cellular features.
- **Visual Reasoning:** Generates real-time "Waterfall Plots" to visualize exactly why a specific patient was classified as Malignant or Benign.

### 3. Enterprise Data Handling

- **Batch Processing:** Supports drag-and-drop `.csv` uploads for bulk inference (500+ patients) with automated result appending.
- **Persistent Storage:** Integrated **SQLite** database for longitudinal patient tracking and audit logs.
- **Automated Reporting:** Generates professional **PDF Medical Reports** containing patient data, diagnostic results, and visual risk analytics.

---

## 📊 Model Performance Metrics

The model was evaluated on a 20% hold-out test set from the UCI WDBC repository.

| Metric        | Score      | Clinical Significance                                          |
| :------------ | :--------- | :------------------------------------------------------------- |
| **Accuracy**  | **98.25%** | Overall correctness of diagnoses.                              |
| **Recall**    | **97.6%**  | Ability to detect actual cancer cases (Low False Negatives).   |
| **Precision** | **98.8%**  | Reliability of a "Malignant" prediction (Low False Positives). |
| **F1-Score**  | **98.2%**  | Balanced harmonic mean of Precision and Recall.                |

---

## 🛠️ Technology Stack

- **Core:** Python 3.9+
- **Web Framework:** Flask (RESTful Architecture)
- **Machine Learning:** Scikit-Learn (Pipelines), XGBoost (Model), SHAP (XAI), NumPy
- **Data Engineering:** Pandas (Batch Processing), SQLAlchemy (ORM), SQLite
- **Frontend:** HTML5, Bootstrap 5, Chart.js (Interactive Visualization)
- **Utilities:** jsPDF (Report Generation), Matplotlib (Static Plotting)

---

## ⚙️ Installation & Setup

Follow these steps to run the application locally.

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/OncoPredict.git](https://github.com/YOUR_USERNAME/OncoPredict.git)
    cd OncoPredict
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**

    ```bash
    python app.py
    ```

4.  **Access the Dashboard**
    Open your browser and navigate to: `http://localhost:5000`

---

## 📂 Project Structure

```text
OncoPredict/
├── app.py                 # Main Flask Application & Logic
├── patients.db            # Local Database (Auto-generated on first run)
├── requirements.txt       # Project Dependencies
├── models/
│   ├── breast_cancer_model.pkl   # Serialized XGBoost Model
│   └── scaler.pkl                # Serialized Data Scaler
├── templates/
│   └── index.html         # Clinical Dashboard UI
└── README.md              # Project Documentation
```
