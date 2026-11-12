# Premier League Match Prediction with ELO

An experimental English Premier League match-outcome prediction project. Its stated approach updates team ELO ratings in chronological order and combines team strength with recent performance. The repository contains several season CSV files, a file named `2 match`, and this README; inspect the data and project file before assuming a runnable script is present.

## Data

The root includes `E0.csv` and the numbered `E0 (...) .csv` files. Confirm each file's season, columns, source, and license before training or redistribution. Do not combine seasons without checking duplicate matches and consistent schemas.

## Getting started

1. Clone the repository and inspect the CSV headers and date ranges.
2. Confirm the contents/type of `2 match` and identify the intended training code.
3. Install Python and create a virtual environment.
4. No root `requirements.txt` or verified run command is currently present; determine dependencies from the actual source before installing.
5. Preserve match chronology when creating rolling features and updating ELO. Split evaluation data by time to avoid future-information leakage.

No test suite or model result is reproduced by these instructions. Record the data versions, ELO parameters, feature definitions, time-based split, and metrics when reporting experiments. Predictions are not betting advice.


## Reproducible run

The extensionless Python script is `2 match`. Run from the repository root; it reads `E0.csv` and `E0 (2).csv` through `E0 (5).csv`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
pip install -r requirements.txt
python "2 match"
```

The script calculates ELO and rolling match-history features; keep the source matches sorted chronologically and do not use future results in features. Confirm season ranges and duplicate matches before updating the CSV files.
