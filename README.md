# Premier League Match Prediction with ELO

An experimental English Premier League outcome classifier using pre-match ELO ratings and recent goal-difference form. Outcomes are home win (H), draw (D), or away win (A).

## Methodology

The maintained `train.py` performs a chronological walk-forward pass:

1. Read and sort the season CSVs by match date.
2. Build each fixture's features from team ratings and the previous five completed matches only.
3. Predict the expected home result using ELO with a configurable home advantage.
4. Update both teams' ratings only after the fixture result is observed.
5. Append goal difference to the rolling history after feature creation.

This order prevents future fixtures from contaminating earlier match features.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

The script reads `E0.csv` and available numbered season files from the repository root.

## Evaluation

Data is split chronologically into train, validation and test periods. Candidate models are selected on validation weighted F1, then refitted on train + validation before one final test evaluation. ELO parameters and metrics are written to `artifacts/metrics.json`; the fitted pipeline is saved alongside it.

## Limitations

ELO parameters and home advantage are configurable research assumptions, not universal constants. Match results are inherently uncertain. Confirm data provenance and avoid treating model outputs as betting advice. No performance score is claimed until reproduced on the current dataset.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
