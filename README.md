# ML_lab_model_2
---
# 🍷 Wine Quality Predictor

**Wine Quality Predictor** is an AI-powered web application that predicts wine quality based on physicochemical properties. Built with **React** for the frontend and **FastAPI** for the backend, it uses a **Decision Tree machine learning pipeline** to classify wines into three categories: **Substandard**, **Satisfactory**, and **Premium**.

---

## 🚀 Features

* **Multi-class classification** using Decision Tree ML
* Predict wine quality from **11 key physicochemical features**
* **Real-time probability analysis** for each quality class
* **Top contributing factors** and expert recommendations
* **Preset wine examples** for quick testing
* Fully **responsive and accessible** UI
* **Batch predictions** support via API
* API metadata and feature importance endpoints for transparency

---

## 🧪 Frontend

**Technology:** React + Lucide React icons + CSS

### Key Functionalities:

* **Dynamic input sliders** for wine features:

  * Acidity & pH: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `pH`
  * Composition: `residual_sugar`, `chlorides`, `density`, `sulphates`
  * Preservatives & alcohol: `free_sulfur_dioxide`, `total_sulfur_dioxide`, `alcohol`
* **Preset wine profiles**: Substandard, Satisfactory, Premium
* **Prediction visualization**:

  * Quality badge
  * Probabilities for each class
  * Top contributing factors
  * Expert recommendations
* **Loading and error handling** with friendly UI
* **Responsive layout** for desktop and mobile

---

## ⚡ Backend

**Technology:** FastAPI + Python + Joblib

### API Endpoints:

| Endpoint                  | Method | Description                            |
| ------------------------- | ------ | -------------------------------------- |
| `/api/predict`            | POST   | Predict quality for a single wine      |
| `/api/predict/batch`      | POST   | Batch prediction for multiple wines    |
| `/api/options`            | GET    | Fetch typical ranges for wine features |
| `/api/feature-importance` | GET    | Retrieve feature importance rankings   |
| `/api/info`               | GET    | Model metadata and statistics          |
| `/api/health`             | GET    | Server health check                    |

### Model

* **Pipeline:** StandardScaler + SMOTE + Decision Tree
* **Classes:** `Substandard`, `Satisfactory`, `Premium`
* **Top Features:** Alcohol, Sulphates, Volatile Acidity

---

## 🛠️ Installation

### Backend

```bash
git clone <repo-url>
cd backend
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

* API will be available at: `http://127.0.0.1:8000`
* API docs at: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

* App will run on `http://localhost:3000`

> ⚠️ Make sure the backend is running before using the frontend.

---

## 🎯 Usage

1. Adjust sliders for wine properties or load a preset example.
2. Click **Predict Wine Quality**.
3. View:

   * Predicted quality
   * Confidence score
   * Probability distribution
   * Top contributing factors
   * Expert recommendation

> For batch predictions, send an array of wine objects to `/api/predict/batch`.

---

## 📊 Example Input (JSON)

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

---

## 📈 Sample Output (JSON)

```json
{
  "quality": "Satisfactory",
  "confidence": 0.78,
  "probabilities": {
    "Premium": 0.12,
    "Satisfactory": 0.78,
    "Substandard": 0.10
  },
  "quality_score_range": "5-6 (Good wine)",
  "recommendation": "🍷 Decent wine with acceptable quality. Suitable for cooking or casual drinking.",
  "top_factors": [
    {"feature": "Alcohol", "value": 9.4, "importance": 0.32},
    {"feature": "Sulphates", "value": 0.56, "importance": 0.18}
  ]
}
```

---

## 🌟 Why This Project

* Demonstrates **full-stack ML deployment**
* Covers **multi-class prediction**, feature importance, and recommendation logic
* Great **educational tool** for Decision Tree models
* Showcases **React + FastAPI integration**

