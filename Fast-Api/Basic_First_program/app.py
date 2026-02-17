# app.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np

app = FastAPI(title="FastAPI ML App")

# Load templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )


@app.post("/predict")
def predict(request: Request, area: float = Form(...)):
    prediction = model.predict(np.array([[area]]))[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": f"₹ {prediction:,.2f}"
        }
    )
