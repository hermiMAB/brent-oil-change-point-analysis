"""
Task 3: Flask Backend
======================
Serves the Brent oil price data, change point model results, and event
data to the React dashboard.

Place this file at:  backend/app.py
Run with:             python app.py   (or: flask --app app run --debug)

Expects these files (relative to project root, adjust DATA_DIR if needed):
    data/BrentOilPrices.csv
    data/brent_oil_events.csv
    data/model_output.json      <- produced by notebooks/02_bayesian_change_point_analysis.py
"""

import json
import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the React dev server (localhost:3000 / 5173) to call this API

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_prices():
    df = pd.read_csv(os.path.join(DATA_DIR, "BrentOilPrices.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    df = df.sort_values("Date")
    return df


def load_events():
    df = pd.read_csv(os.path.join(DATA_DIR, "brent_oil_events.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_model_output():
    path = os.path.join(DATA_DIR, "model_output.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/prices")
def prices():
    """
    Historical price series.
    Optional query params: start=YYYY-MM-DD, end=YYYY-MM-DD
    """
    df = load_prices()
    start = request.args.get("start")
    end = request.args.get("end")
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    data = [
        {"date": d.strftime("%Y-%m-%d"), "price": p}
        for d, p in zip(df["Date"], df["Price"])
    ]
    return jsonify(data)


@app.route("/api/change-points")
def change_points():
    """Bayesian change point model results (from PyMC notebook output)."""
    result = load_model_output()
    if result is None:
        return jsonify({"error": "Model output not found. Run the Task 2 notebook first."}), 404
    return jsonify(result)


@app.route("/api/events")
def events():
    """Structured events dataset."""
    df = load_events()
    start = request.args.get("start")
    end = request.args.get("end")
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    data = df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    return jsonify(data)


@app.route("/api/event-correlation")
def event_correlation():
    """
    Combines the detected change point with the researched events list,
    ranked by proximity in days, for the dashboard's correlation view.
    """
    model_output = load_model_output()
    events_df = load_events()

    if model_output is None:
        return jsonify({"error": "Model output not found. Run the Task 2 notebook first."}), 404

    change_date = pd.to_datetime(model_output["change_point_date"])
    events_df["days_from_change_point"] = (events_df["Date"] - change_date).dt.days
    events_df = events_df.sort_values("days_from_change_point", key=abs)

    data = events_df.assign(
        Date=events_df["Date"].dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")

    return jsonify({
        "change_point_date": model_output["change_point_date"],
        "avg_price_before_usd": model_output["avg_price_before_usd"],
        "avg_price_after_usd": model_output["avg_price_after_usd"],
        "price_pct_change": model_output["price_pct_change"],
        "ranked_events": data,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
