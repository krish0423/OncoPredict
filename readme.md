# OncoPredict: XGBoost Clinical Diagnostic System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-black?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable-AI-green?style=for-the-badge)

**OncoPredict** is an enterprise-grade medical AI application capable of diagnosing breast cancer malignancy with **98.2% accuracy**. It integrates a Gradient Boosting classifier with **SHAP (Shapley Additive exPlanations)** to provide transparent, feature-level reasoning for every diagnosis, bridging the gap between "Black Box" AI and clinical trust.

---

## 🚀 Key Technical Features

### 1. High-Performance Inference Engine

- **Model:** XGBoost Classifier trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
- **Performance:** Optimized for high Recall (Sensitivity) to minimize false negatives in medical contexts.
- **Preprocessing:** Robust scaling pipelines to handle outlier variations in cellular data.

### 2. Explainable AI (XAI) Architecture

- Integrates **SHAP** to calculate the marginal contribution of all 30 cellular features.
- Generates real-time "Waterfall Plots" to visualize why a specific patient was classified as Malignant or Benign.

### 3. Enterprise Data Handling

- **Batch Processing:** Supports `.csv` uploads for bulk inference (500+ patients) with automated result appending.
- **Persistent Storage:** SQLite database integration for longitudinal patient tracking and audit logs.
- **Automated Reporting:** Generates comprehensive PDF medical reports including risk probability and feature deviations.

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
- **Web Framework:** Flask (RESTful API)
- **ML Libraries:** Scikit-Learn, XGBoost, SHAP, NumPy, Pandas
- **Visualization:** Matplotlib (Static generation), Chart.js (Frontend interactivity)
- **Database:** SQLAlchemy ORM (SQLite)

---

## 💻 API Usage Example

The system exposes REST endpoints for integration with hospital systems.

**Endpoint:** `POST /predict`
**Content-Type:** `application/json`

**Request Payload:**

```json
{
  "data": [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,
    0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
    0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189
  ]
}
```
