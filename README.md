# Movie Revenue Prediction

A machine‑learning study that models theatrical box‑office revenue using rich metadata from the **TMDB 5000 Movie** corpus.
The pipeline spans rigorous data cleansing, domain‑driven feature construction, and comparative evaluation of three regression families—linear, instance‑based, and ensemble—under repeated cross‑validation.
All experiments are fully reproducible from the supplied notebook and script.

---

## Data

| File                         | Description                                                  | Rows   | Size    |
| ---------------------------- | ------------------------------------------------------------ | ------ | ------- |
| `data/tmdb_5000_movies.csv`  | Core movie attributes (budget, revenue, genres, dates, etc.) |  4 803 |  5.4 MB |
| `data/tmdb_5000_credits.csv` | Cast and crew JSON blobs keyed by `movie_id`                 |  4 803 |  40 MB  |

### Target

`revenue` – worldwide theatrical gross in USD.

### Core explanatory fields

Budget, popularity index, runtime, release timestamp, nested JSON for genres, production companies, collections, cast, crew, languages.

---

## Repository Layout

```
movie-revenue-prediction/
├── data/                        # raw CSVs (TMDB export)
├── notebooks/
│   └── movie_revenue_prediction.ipynb   # end‑to‑end workflow
├── scripts/
│   └── feature_engineering_and_modeling.py
├── report/
│   └── movie_revenue_prediction_report.pdf
├── requirements.txt
└── README.md
```

---

## Environment

```
python 3.10+
pandas  2.2
numpy   2.x
scikit-learn 1.5
matplotlib 3.9
```

Install:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Data exploration / training
jupyter lab notebooks/movie_revenue_prediction.ipynb

# 2. Headless run (recreates final metrics):
python scripts/feature_engineering_and_modeling.py
```

The script regenerates train/test splits, fits all models, prints validation metrics, and serialises artefacts to `outputs/`.

---

## Methodology

### 1  Data Cleaning

* Removed implausible entries using compound thresholds:
  – `budget ≤ 175 M`, `revenue ≤ 700 M`,
  – `vote_count ≤ 8 000`, `3.5 ≤ vote_average ≤ 8.3`,
  – `60 ≤ runtime ≤ 200`, `popularity ≤ 150`.
* Synchronised `credits` to filtered movie set; resulted in **4 504** observations.

### 2  Feature Engineering

* **Genres**: binary indicators for genres with frequency ≥ 50; residual category `genre_other`.
* **Temporal**: month, weekday, summer/holiday/spring‑break flags, five‑year bins (`period_1990‑1994`, …).
* **Studios**: flags for major distributors; per‑studio historical mean revenue.
* **Franchise**: boolean `is_franchise` from `belongs_to_collection`; collection‑level mean revenue.
* **Talent prestige**:
  – `is_famous_director`, `director_avg_revenue` (≥ 5 high‑impact releases),
  – `famous_actor_count` (cast overlap with ≥ 5 high‑impact titles).
* **Interactions**: 12 multiplicative terms capturing effect modulation (e.g., `budget × popularity`, `franchise × budget`).
* **Transformations**: `log1p` on budget/popularity, quadratic terms for key numerics.

Final design matrix: **\~100 numeric features** after one‑hot encoding.

### 3  Learning Algorithms

| Model                       | Hyper‑parameters                                                    | Fit objective                       |
| --------------------------- | ------------------------------------------------------------------- | ----------------------------------- |
| **Ordinary Least Squares**  | closed form                                                         | minimise RSS under full column rank |
| **k‑Nearest Neighbours**    | `k=5`, Euclidean, features z‑scored                                 | local average prediction            |
| **Random Forest Regressor** | 200 trees, `max_depth=25`, `max_features=√p`, `min_samples_split=5` | bootstrap aggregation of CART       |

Hyper‑parameters chosen via grid search on training folds (5‑fold K‑Fold, `random_state=42`).

### 4  Validation Protocol

* **Hold‑out split**: 80 % train / 20 % test, stratified on revenue quartiles.
* **Cross‑validation**: 5‑fold repeated on full dataset for out‑of‑sample generalisation estimate.
* **Metrics**: MAE, RMSE, R², Residual Standard Error (bias‑adjusted RMSE).

---

## Results

| Model                           | MAE (USD)  | RMSE (USD) | R²    | Notes                             |
| ------------------------------- | ---------- | ---------- | ----- | --------------------------------- |
| Linear Regression               | 36.1 M     | 56.6 M     | 0.769 | strong linear signal captured     |
| k‑NN `k=5`                      | 41.3 M     | 73.3 M     | 0.613 | suffers in high‑dim space         |
| Random Forest (top‑20 features) | **31.4 M** | 58.7 M     | 0.752 | best MAE; robust to non‑linearity |

*Random Forest attains the lowest absolute error while preserving interpretability through feature importance.*

---

## Reproducibility

* Deterministic random seeds in scikit‑learn (`random_state=42`).
* Environment versions fixed in `requirements.txt`.
* Notebook outputs cleared; reruns regenerate identical metrics.

---

## Authors

* Fardin Iqbal — model development, evaluation
* Yetro Cheng — feature engineering, experimental design
* Tanjim Ahammad — data cleaning, preprocessing

---

## License

MIT License (see `LICENSE`).
