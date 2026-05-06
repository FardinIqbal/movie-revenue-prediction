# Architectural Decision Records

Each entry documents a non-obvious modeling or feature-engineering choice.

---

## ADR 1: Compound outlier filtering instead of robust regression

**Status:** accepted

**Context:**
Box-office revenue is heavy-tailed. A few mega-hits dominate the dataset. Two approaches:

1. **Robust regression**: use a model less sensitive to outliers (e.g., Huber regression, RANSAC).
2. **Compound outlier filtering** (this design): explicit thresholds on budget, revenue, vote_count, vote_average, runtime, and popularity. Films outside the bands are removed before training.

**Decision:**
Compound filtering.

**Consequences:**

- Pro: makes the modeling problem well-conditioned. Mainstream films behave predictably; mega-hits and art-house films do not.
- Pro: keeps OLS competitive. Robust regression sacrifices efficiency on the typical case.
- Pro: explicit. The thresholds are visible in the source and the report.
- Con: removes ~6% of the data. Information is lost.
- Con: thresholds are pragmatic, not principled. Different choices yield different removal rates.

For a study on mainstream film economics, filtering is the right call. For a model intended to predict every film including mega-hits, robust regression would be a better fit.

---

## ADR 2: "Famous director/actor" flag at ≥5 high-impact films

**Status:** accepted

**Context:**
Cast and crew prestige are predictive of revenue, but raw counts (number of cast members, number of crew) are not (every film has slots filled). Two options:

1. **Use raw counts**: cast size, crew size.
2. **Define a prestige threshold** (this design): a director or actor is "famous" if they appear in ≥5 films with budget > $50M and vote-average top quartile.

**Decision:**
Prestige threshold.

**Consequences:**

- Pro: captures A-list status. Tom Cruise has many high-impact films; an extra in twenty Z-budget films does not.
- Pro: binary flag is interpretable.
- Pro: works for both directors and actors with the same definition.
- Con: arbitrary threshold. Why 5 films and not 3 or 10? Why $50M budget? The thresholds are based on visual inspection of the distribution.
- Con: time-static. A director's prestige today may differ from their prestige in 1990.

The threshold is a judgment call. Sensitivity analysis shows the model performance is stable for thresholds 3-7 films. For a production model, replacing this with a continuous "prestige score" derived from external data (Oscar nominations, Letterboxd ratings) would be the upgrade.

---

## ADR 3: Random Forest as the primary model

**Status:** accepted

**Context:**
Three model families considered: OLS, kNN, Random Forest. Each has tradeoffs.

**Decision:**
Random Forest as primary; OLS and kNN as comparison points.

**Consequences:**

- Pro: best MAE (31.4M vs 36.1M for OLS, 41.3M for kNN).
- Pro: handles non-linear effects and interactions automatically (without needing manual interaction terms, though we added them anyway).
- Pro: robust to outliers within the filtered dataset.
- Pro: feature importance is a useful by-product.
- Con: less interpretable than OLS. Cannot point to a single coefficient and say "budget contributes X."
- Con: slower to train. ~30 seconds for 200 trees on 4,504 rows.

For a study comparing predictive power, RF is the right primary model. For a production deployment where interpretability matters, OLS would be the better choice (with the cost of higher MAE).

---

## ADR 4: Twelve interaction terms, hand-selected

**Status:** accepted

**Context:**
Multiplicative interactions encode effect modulation that linear features miss. Two options:

1. **All pairwise interactions**: combinatorial explosion (~5,000 features for our base set).
2. **Hand-selected interactions** (this design): 12 specific multiplicative terms based on domain knowledge.

**Decision:**
Hand-selected.

**Consequences:**

- Pro: bounded feature count. ~100 features total instead of thousands.
- Pro: each interaction has a domain rationale (franchise × budget, popularity × budget, etc.).
- Pro: avoids the curse of dimensionality on the kNN model.
- Con: misses interactions that domain knowledge does not predict.
- Con: each interaction is a manual decision; reproducing requires replicating the same choices.

For a tractable feature space, hand-selection is the right call. A pure ML approach would use feature selection or regularization to handle all pairwise interactions.

---

## ADR 5: Median normalization in feature engineering

**Status:** N/A

(This project does not normalize features the way GLIMPSE does. Both linear and tree-based models work on the raw feature scale; tree models are scale-invariant, OLS handles scale via coefficients. Mentioned only because the same author wrote both projects, and "did you remember to normalize?" is a common reviewer question.)

---

## ADR 6: 5-fold cross-validation on full dataset, plus 80/20 holdout

**Status:** accepted

**Context:**
Validating model performance. Two evaluation strategies were considered:

1. **80/20 holdout only**: train on 80%, evaluate on 20%, report metrics.
2. **K-fold cross-validation only**: split into 5 folds, train on 4, test on 1, repeat.
3. **Both** (this design): 80/20 holdout for the headline number; 5-fold CV for robustness.

**Decision:**
Both.

**Consequences:**

- Pro: 80/20 gives a single, citable metric.
- Pro: 5-fold CV gives a robustness check. If the headline metric matches the CV average, the model is not just lucky on one split.
- Pro: stratification on revenue quartiles ensures the test set has the same revenue distribution as the train set.
- Con: more compute. CV trains the model 5 times.

Worth the cost. A model whose 80/20 metric differs significantly from the CV average is overfitting; we want to catch that.

---

## ADR 7: All work in a single Jupyter notebook plus a single Python script

**Status:** accepted

**Context:**
Project organization. Two options:

1. **Notebook only**: all work in `movie_revenue_prediction.ipynb`.
2. **Notebook + script** (this design): notebook for exploration, script (`feature_engineering_and_modeling.py`) for batch reproducibility.

**Decision:**
Both.

**Consequences:**

- Pro: notebook is interactive (good for exploration, presentation).
- Pro: script is automatable (good for CI, reproducibility).
- Pro: side-by-side, the notebook documents *why*, the script documents *what*.
- Con: two artifacts to keep in sync. If the notebook is updated, the script must be regenerated.

For an academic study with a written report, the notebook+script combo is the right deliverable. Production deployments would use the script only, possibly converted to a more structured pipeline (Airflow, dbt, etc.).
