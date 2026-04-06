"""
Mall Customer Segmentation API
FastAPI backend — also serves the frontend at http://localhost:8000/app
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle, os, pathlib

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mall Customer Segmentation API",
    description="KMeans clustering on Mall Customers dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve frontend ───────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent

@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>index.html not found next to main.py</h2>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# ─── Cluster labels ───────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: {"name": "Careful Spenders",   "color": "#7c3aed", "emoji": "🛡️",
        "description": "Middle-aged, moderate income, low spending. Cautious buyers."},
    1: {"name": "High-Value Targets", "color": "#ef4444", "emoji": "💎",
        "description": "Young, high income, high spending. Premium shoppers."},
    2: {"name": "Budget Conscious",   "color": "#f97316", "emoji": "💰",
        "description": "Young, low income, high spending. Impulsive buyers on a budget."},
    3: {"name": "Conservative Elite", "color": "#22c55e", "emoji": "🎯",
        "description": "Older, high income, low spending. Selective, quality-focused."},
    4: {"name": "Average Joes",       "color": "#3b82f6", "emoji": "🏪",
        "description": "Average on all metrics. Mainstream shoppers."},
}

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / "kmeans_model.pkl"
DATA_PATH  = BASE_DIR / "Mall_Customers.csv"

def train_and_save():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"'{DATA_PATH}' not found. Place Mall_Customers.csv next to main.py.")
    df = pd.read_csv(DATA_PATH)
    X  = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    km = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
    km.fit(X)
    df["label"] = km.predict(X).astype(int)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": km, "df": df}, f)
    return km, df

def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        return obj["model"], obj["df"]
    return train_and_save()

km_model, customers_df = load_model()

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:            float = Field(..., ge=18, le=100, example=23)
    annual_income:  float = Field(..., ge=0,  le=300, example=60,  alias="annual_income_k")
    spending_score: float = Field(..., ge=1,  le=100, example=75)

    class Config:
        populate_by_name = True

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "API running 🛍️  →  open /app for the dashboard, /docs for API explorer"}

@app.post("/predict", tags=["Prediction"])
def predict_segment(req: PredictRequest):
    features   = np.array([[req.age, req.annual_income, req.spending_score]])
    cluster_id = int(km_model.predict(features)[0])
    info       = CLUSTER_LABELS[cluster_id]
    return {
        "cluster_id":  cluster_id,
        "name":        info["name"],
        "description": info["description"],
        "color":       info["color"],
        "emoji":       info["emoji"],
    }     

@app.get("/clusters", tags=["Analytics"])
def get_clusters():
    result = []
    for cid in range(5):
        grp  = customers_df[customers_df["label"] == cid]
        info = CLUSTER_LABELS[cid]
        result.append({
            "cluster_id":   cid,
            "name":         info["name"],
            "description":  info["description"],
            "color":        info["color"],
            "emoji":        info["emoji"],
            "count":        len(grp),
            "avg_age":      round(float(grp["Age"].mean()), 1),
            "avg_income":   round(float(grp["Annual Income (k$)"].mean()), 1),
            "avg_spending": round(float(grp["Spending Score (1-100)"].mean()), 1),
        })
    return result

@app.get("/customers", tags=["Data"])
def get_customers(cluster_id: int = None):
    df = customers_df
    if cluster_id is not None:
        if cluster_id not in range(5):
            raise HTTPException(400, "cluster_id must be 0-4")
        df = df[df["label"] == cluster_id]
    records = df[["CustomerID", "Gender", "Age",
                  "Annual Income (k$)", "Spending Score (1-100)", "label"]].to_dict("records")
    for r in records:
        lbl = int(r["label"])
        r["label"]        = lbl
        r["cluster_name"] = CLUSTER_LABELS[lbl]["name"]
        r["color"]        = CLUSTER_LABELS[lbl]["color"]
    return {"total": len(records), "customers": records}

@app.get("/stats", tags=["Analytics"])
def get_stats():
    df = customers_df
    return {
        "total_customers": len(df),
        "gender_split":    df["Gender"].value_counts().to_dict(),
        "age_range":       {"min": int(df["Age"].min()), "max": int(df["Age"].max()), "mean": round(float(df["Age"].mean()), 1)},
        "income_range":    {"min": int(df["Annual Income (k$)"].min()), "max": int(df["Annual Income (k$)"].max()), "mean": round(float(df["Annual Income (k$)"].mean()), 1)},
        "spending_range":  {"min": int(df["Spending Score (1-100)"].min()), "max": int(df["Spending Score (1-100)"].max()), "mean": round(float(df["Spending Score (1-100)"].mean()), 1)},
        "clusters": 5,
    }



















