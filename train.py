from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FILES = ["E0.csv", "E0 (2).csv", "E0 (3).csv", "E0 (4).csv", "E0 (5).csv"]
OUTPUT_DIR = ROOT / "artifacts"
SEED, INITIAL_ELO, K_FACTOR, HOME_ADVANTAGE = 42, 1500.0, 20.0, 50.0


def load_matches():
    frames = []
    for name in FILES:
        path = ROOT / name
        if path.is_file():
            frame = pd.read_csv(path)
            if "Div" in frame:
                frame = frame.loc[frame["Div"] == "E0"]
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No E0 season CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ("FTHG", "FTAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df["FTR"].isin(["H", "D", "A"])]
    return df.drop_duplicates(["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)


def build_pre_match_features(matches):
    teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    elo = {team: INITIAL_ELO for team in teams}
    goal_diff = defaultdict(lambda: deque(maxlen=5))
    rows = []
    for _, match in matches.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        home_elo, away_elo = elo[home], elo[away]
        home_form = float(np.mean(goal_diff[home])) if goal_diff[home] else 0.0
        away_form = float(np.mean(goal_diff[away])) if goal_diff[away] else 0.0
        rows.append({
            "HomeELO": home_elo, "AwayELO": away_elo,
            "ELO_Diff": home_elo - away_elo + HOME_ADVANTAGE,
            "Home_GoalDiff_Last5": home_form,
            "Away_GoalDiff_Last5": away_form,
            "GoalDiff_Diff": home_form - away_form,
            "Target": match["FTR"],
        })

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400.0))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[match["FTR"]]
        elo[home] = home_elo + K_FACTOR * (actual_home - expected_home)
        elo[away] = away_elo + K_FACTOR * ((1.0 - actual_home) - (1.0 - expected_home))
        hg, ag = float(match["FTHG"]), float(match["FTAG"])
        goal_diff[home].append(hg - ag)
        goal_diff[away].append(ag - hg)
    return pd.DataFrame(rows)


def main():
    matches = load_matches()
    data = build_pre_match_features(matches)
    if len(data) < 60:
        raise ValueError(f"Only {len(data)} rows available; at least 60 are required.")
    X, y = data.drop(columns=["Target"]), data["Target"]
    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_trainval, y_trainval = X.iloc[:val_end], y.iloc[:val_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=SEED),
    }
    ranked = []
    for name, estimator in candidates.items():
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        ranked.append((f1_score(y_val, pred, average="weighted", zero_division=0), name, estimator))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_val_f1, best_name, best_estimator = ranked[0]
    final_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", best_estimator)])
    final_model.fit(X_trainval, y_trainval)
    test_pred = final_model.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "validation_weighted_f1": float(best_val_f1),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_weighted_f1": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_rows": int(len(y_test)),
        "feature_rows": int(len(data)),
        "date_start": str(matches["Date"].min().date()),
        "date_end": str(matches["Date"].max().date()),
        "elo": {"initial": INITIAL_ELO, "k_factor": K_FACTOR, "home_advantage": HOME_ADVANTAGE},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, OUTPUT_DIR / "elo_match_model.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_pred, labels=["H", "D", "A"], zero_division=0))


if __name__ == "__main__":
    main()
