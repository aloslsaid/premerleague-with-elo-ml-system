# Premier League ELO + Machine Learning Predictor

> A football match prediction experiment combining dynamic ELO team ratings with recent-form features and supervised classification.

## Concept
Raw league tables and season totals can hide how team strength changes over time. This project maintains an ELO rating for each team and updates it after every completed fixture. The current ratings, together with recent goal-difference form, become inputs to machine-learning models that estimate match outcomes.

## How the pipeline works
1. Load and combine the historical Premier League season files.
2. Keep the date, home team, away team, full-time goals, and result fields required by the rating engine.
3. Parse dates and sort matches chronologically.
4. Initialize team ratings at 1500.
5. Calculate expected home-win probability from the rating difference using the standard ELO logistic formula.
6. Compare the expected result with the actual result and update both teams' ratings using a K-factor.
7. Track each team's recent match history and calculate rolling goal-difference features from the previous five matches.
8. Build a match dataset from pre-match information and train classification models.
9. Compare predictions using Accuracy, Precision, Recall, F1-score, and a confusion matrix.

## Why ELO?
ELO provides a compact, continuously updated estimate of relative team strength. It is complementary to recent-form features: ratings summarize longer-term strength, while recent goal difference captures a shorter-term performance signal.

## Models
The script imports Scikit-learn classifiers including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, and SVM. The experiment compares their classification performance rather than assuming one algorithm is universally best.

## Installation
Python 3.10+ is recommended.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn
```

## Run
```bash
python "2 match"
```
Run from the repository root. The script expects `E0.csv` and `E0 (2).csv` through `E0 (5).csv` in that directory.

## Validation discipline
The ELO rating and rolling statistics must be updated only after a match has been represented using pre-match features; otherwise, future information can leak into training. Treat all scores as historical experimental results, not a guarantee of future match outcomes or betting returns.

## Repository contents
- `2 match` — ELO calculation, feature generation, training, and evaluation.
- `E0*.csv` — historical Premier League match data.

## Skills demonstrated
ELO rating systems · Sequential feature engineering · Football analytics · Multi-class classification · Model comparison · Confusion-matrix analysis
