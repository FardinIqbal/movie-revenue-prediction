# Box Office Revenue Predictor

End-to-end ML pipeline predicting box-office revenue from TMDB 5000 metadata. Three regression families compared under 5-fold cross-validation. Random Forest wins on MAE at $31.4M; Linear Regression wins on R-squared at 0.769.

```
Dataset         TMDB 5000 Movies + Credits  (4,803 rows -> 4,504 after cleaning)
Features        ~100 engineered numeric features
Models          OLS, k-NN (k=5), Random Forest (200 trees, depth 25)
Best MAE        $31.4M  (Random Forest, top-20 features)
Best R^2        0.769   (Linear Regression)
Validation      80/20 hold-out + 5-fold K-Fold (random_state=42)
```

---

## What it is

A graduate-level machine learning study modeling worldwide theatrical revenue from movie metadata. The pipeline covers outlier filtering, nested-JSON extraction, domain-driven feature construction (talent prestige, franchise effects, temporal windows, studio history), and head-to-head evaluation of linear, instance-based, and ensemble regressors on the same design matrix.

Repository contents:

```
data/                             raw TMDB CSVs (movies + credits)
notebooks/
  movie_revenue_prediction.ipynb  end-to-end workflow (EDA, FE, modeling)
scripts/
  feature_engineering_and_modeling.py  headless reproduction of final metrics
report/
  movie_revenue_prediction_report.pdf  written findings
```

---

## Dataset

TMDB 5000 Movie Dataset (Kaggle mirror of TMDb public metadata).

| File | Rows | Size | Contents |
|---|---|---|---|
| `data/tmdb_5000_movies.csv` | 4,803 | 5.4 MB | budget, revenue, genres, dates, popularity, runtime, votes, nested JSON for collections/companies/languages |
| `data/tmdb_5000_credits.csv` | 4,803 | 40 MB | cast and crew as JSON blobs keyed by `movie_id` |

**Target:** `revenue` (worldwide theatrical gross, USD).

**Cleaning filters** (compound thresholds on implausible extremes):

- `budget <= 175M`, `revenue <= 700M`
- `vote_count <= 8,000`, `3.5 <= vote_average <= 8.3`
- `60 <= runtime <= 200`, `popularity <= 150`

After filtering: **4,504 observations**. Credits joined on the filtered movie set.

---

## Feature engineering

The final design matrix contains **~100 numeric features** derived from the raw metadata:

| Group | Features |
|---|---|
| Genres | binary indicators for genres with frequency >= 50, plus `genre_other` residual |
| Temporal | release month, weekday, summer/holiday/spring-break flags, five-year period bins (`period_1990_1994`, ...) |
| Studios | flags for major distributors; per-studio historical mean revenue |
| Franchise | boolean `is_franchise` from `belongs_to_collection`; collection-level mean revenue |
| Talent prestige | `is_famous_director`, `director_avg_revenue` (directors with >=5 high-impact releases); `famous_actor_count` (cast overlap with >=5 high-impact titles) |
| Interactions | 12 multiplicative terms (`budget x popularity`, `franchise x budget`, etc.) |
| Transformations | `log1p` on budget and popularity; quadratic terms on key numerics |

All categorical fields are one-hot encoded. Numeric features are z-scored for k-NN; left raw for OLS and Random Forest.

---

## Models compared

| Model | Hyperparameters | Fit objective |
|---|---|---|
| Ordinary Least Squares | closed-form | minimize RSS under full column rank |
| k-Nearest Neighbors | `k=5`, Euclidean, z-scored features | local mean prediction |
| Random Forest Regressor | 200 trees, `max_depth=25`, `max_features=sqrt(p)`, `min_samples_split=5` | bootstrap aggregation of CART |

Hyperparameters selected via grid search on training folds. All models share the same cleaned dataset and design matrix.

---

## Evaluation

- **Hold-out split:** 80% train / 20% test, `random_state=42`.
- **Cross-validation:** 5-fold K-Fold on the full dataset (`KFold(n_splits=5, shuffle=True, random_state=42)`), `cross_val_score` per model.
- **Metrics:** MAE, RMSE, R-squared. Random Forest also evaluated on the top-20 features by importance.

### Results

| Model | MAE (USD) | RMSE (USD) | R^2 | Notes |
|---|---|---|---|---|
| Linear Regression | 36.1M | 56.6M | **0.769** | captures strong linear signal; best R-squared |
| k-NN (k=5) | 41.3M | 73.3M | 0.613 | degrades in high-dim feature space |
| Random Forest (top-20 features) | **31.4M** | 58.7M | 0.752 | best MAE; robust to non-linearity |

Random Forest minimizes absolute error while exposing feature importances; OLS explains the most variance but overshoots on long-tail blockbusters.

---

## Build and run

**Requirements:** Python 3.10+, pandas 2.2, numpy 2.x, scikit-learn 1.5, matplotlib 3.9.

```bash
git clone https://github.com/FardinIqbal/movie-revenue-prediction.git
cd movie-revenue-prediction

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "pandas>=2.2" "numpy>=2.0" "scikit-learn>=1.5" "matplotlib>=3.9" jupyter
```

Reproduce results:

```bash
# Option A: interactive notebook (EDA + modeling)
jupyter lab notebooks/movie_revenue_prediction.ipynb

# Option B: headless script (regenerates final metrics)
python scripts/feature_engineering_and_modeling.py
```

Determinism: all random seeds fixed at `random_state=42`. Reruns reproduce identical metrics.

---

## Tech stack

Python, scikit-learn, Pandas, NumPy, matplotlib, Jupyter.

---

## Academic context

Stony Brook University graduate ML coursework, April-May 2025. Team project (Fardin Iqbal, Yetro Cheng, Tanjim Ahammad) covering EDA, feature engineering, modeling, and cross-validated evaluation against a written report and live demo.

Contributions:
- **Fardin Iqbal** - model development, evaluation, cross-validation
- **Yetro Cheng** - feature engineering, experimental design
- **Tanjim Ahammad** - data cleaning, preprocessing

---

## License

MIT License. See [`LICENSE`](LICENSE).
