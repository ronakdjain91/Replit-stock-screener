import os
import io
import glob
import threading
import subprocess
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request
from datetime import datetime

app = Flask(__name__)

scan_status = {"main": "idle", "nifty": "idle"}


def run_main_scan():
    scan_status["main"] = "running"
    try:
        subprocess.run(["python", "main.py"], capture_output=True, timeout=600)
        scan_status["main"] = "done"
    except Exception as e:
        scan_status["main"] = f"error: {str(e)}"


def run_nifty_scan():
    scan_status["nifty"] = "running"
    try:
        subprocess.run(["python", "2nd.py"], capture_output=True, timeout=600)
        scan_status["nifty"] = "done"
    except Exception as e:
        scan_status["nifty"] = f"error: {str(e)}"


def get_latest_nifty_csv():
    files = glob.glob("nifty_scan_*.csv")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data/main")
def data_main():
    path = "nifty50_signals.csv"
    if not os.path.exists(path):
        return jsonify({"columns": [], "rows": [], "generated_at": None})
    df = pd.read_csv(path)
    df = df.round(4)
    cols = list(df.columns)
    rows = df.fillna("").values.tolist()
    mtime = os.path.getmtime(path)
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
    return jsonify({"columns": cols, "rows": rows, "generated_at": generated_at})


@app.route("/api/data/nifty")
def data_nifty():
    path = get_latest_nifty_csv()
    if not path:
        return jsonify({"columns": [], "rows": [], "generated_at": None})
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if "meta" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.round(4)
    cols = list(df.columns)
    rows = df.fillna("").values.tolist()
    mtime = os.path.getmtime(path)
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
    return jsonify({"columns": cols, "rows": rows, "generated_at": generated_at})


@app.route("/api/run/<scanner>", methods=["POST"])
def run_scanner(scanner):
    if scanner == "main" and scan_status["main"] != "running":
        t = threading.Thread(target=run_main_scan)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    elif scanner == "nifty" and scan_status["nifty"] != "running":
        t = threading.Thread(target=run_nifty_scan)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/status/<scanner>")
def status(scanner):
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/download/main")
def download_main():
    path = "nifty50_signals.csv"
    if not os.path.exists(path):
        return jsonify({"error": "No file found"}), 404
    return send_file(path, as_attachment=True, download_name="nifty100_signals.csv")


@app.route("/api/download/nifty")
def download_nifty():
    path = get_latest_nifty_csv()
    if not path:
        return jsonify({"error": "No file found"}), 404
    return send_file(path, as_attachment=True, download_name="nifty50_scan.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
