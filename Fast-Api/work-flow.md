# 🚀 FastAPI + Machine Learning Web App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-⚡-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Model-orange)
![Status](https://img.shields.io/badge/Status-Working-success)

A beginner-friendly project demonstrating how to **train a Machine Learning model once**, save it using **pickle**, and reuse it efficiently inside a **FastAPI-powered website**.

---

## 📂 Project Structure


Fast_ML_App/
│
├── app.py
├── model.py
├── model.pkl
├── requirements.txt
│
├── templates/
│ └── index.html
│
└── static/
└── css/
└── style.css


---

## 🎯 What This Project Does

- Trains an ML model once
- Saves the trained model using `pickle`
- Loads the model into FastAPI
- Accepts user input from a website
- Returns predictions instantly

No retraining. No delays. Just fast predictions ⚡

---

## 🔄 Application Flow (Visualized)

```mermaid
flowchart LR
    A[User Browser 🌐] --> B[FastAPI Server ⚡]
    B --> C[Loaded ML Model 🧠]
    C --> D[Prediction Result 📊]
    D --> A
