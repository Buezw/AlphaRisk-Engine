from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd

app = FastAPI(title="AlphaRisk API")

# Configure CORS so the React frontend can reach this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins in the dev environment
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    conn = sqlite3.connect("factor_risk.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/risk-summary")
def get_risk_summary():
    """
    Pulls the latest factor-exposure snapshot from the database and computes the aggregate risk share.
    """
    conn = get_db_connection()

    # Pull the latest beta loadings for every asset
    df = pd.read_sql_query("SELECT * FROM factor_exposures", conn)
    conn.close()

    if df.empty:
        return {"status": "error", "message": "No data found"}

    # Assemble the JSON payload for the frontend
    exposures = df[['symbol', 'beta_mkt', 'beta_smb', 'beta_hml']].to_dict(orient="records")

    # A quick stand-in aggregation of total risk (in production this should call the
    # RiskDecompositionEngine from earlier) — kept as a simple mean for now just to get the
    # frontend/backend integration working end-to-end
    avg_r2 = df['r_squared'].mean()

    return {
        "status": "success",
        "data": {
            "kpi": {
                "total_variance": 0.000105,  # placeholder example value
                "systematic_ratio": avg_r2,
                "idiosyncratic_ratio": 1.0 - avg_r2
            },
            "exposures": exposures
        }
    }