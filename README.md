# Europred

**From historical football results to Bayesian score distributions and match probabilities.**

I first implemented this model in 2024, in 11th grade. It was my first experience using scientific Python libraries to apply a substantial statistical model to a practical problem. I later revisited the model in ForecastDIY, applying what I had learned in computer science to its organization and integration into a broader prediction platform.

This standalone edition was prepared with AI assistance to make the earlier model portable and easy to review. It uses explicit CSV inputs and synthetic demo data in place of machine-specific paths and private file dependencies.

## Start here

- **Model:** [`fit_goal_model`](europred.py) translates observed goals and opponent ratings into a negative-binomial posterior predictive distribution.
- **Prediction:** `predict_fixture` combines the two teams' goal distributions into a joint score matrix and win/draw/loss probabilities.
- **Complete example:** the command below reads three CSVs, selects only earlier matches, predicts the fixtures, and exports the results.

## Run

Tested with Python 3.13.5, NumPy 2.2.1, pandas 2.2.3, and SciPy 1.15.1. Dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python europred.py --output outputs
```

The program writes `predictions.csv` and one `score_matrix_001.csv`-style file per fixture, numbered in input order. Choose a fresh output directory for each run to keep results together. All probabilities are in **[0, 1]**, not percentages. Score-matrix rows are home goals and columns are away goals. The full retained matrix is exported, without display rounding affecting the calculation.

To use your own data:

```bash
python europred.py \
  --history data/history.csv \
  --ratings data/ratings.csv \
  --fixtures data/fixtures.csv \
  --output my_predictions
```

All bundled team names, ratings and results are fictional. The data demonstrate the calculation, not predictive accuracy or a historical tournament result. No network access, account, API key, or original private data is needed.

## Data to prediction

1. Load historical results, a declared ratings snapshot, and the fixtures to predict.
2. Validate names, dates, ratings and scores; reject duplicate matches and self-matches.
3. For each fixture, select only results with `Date` **strictly before** its date.
4. Gather each team's goals and historical opponents' ratings, then calculate its posterior predictive distribution against the new opponent.
5. Form the independent joint score distribution and sum its lower triangle, diagonal and upper triangle for home win, draw and away win.
6. Export the probabilities, expected goals, usable history counts and omitted tail probability.

Input schemas:

| File | Required columns |
|---|---|
| Historical results | `Date,HomeTeam,AwayTeam,HomeScore,AwayScore` |
| Ratings snapshot | `Name,Rating,AsOf` |
| Fixtures | `Date,HomeTeam,AwayTeam` |

Dates use `YYYY-MM-DD`. Scores are nonnegative integers; ratings are finite positive numbers. `AsOf` states when a rating was available and must precede the fixture date. Team names must match across files. At most 25,000 rows per input, 5 MiB per file and 50 fixtures per run are supported.

## The Bayesian model

The statistical method is based on **Suzuki, A. K., Salasar, L. E. B., Leite, J. G., and Louzada-Neto, F. (2010). [A Bayesian approach for predicting match outcomes: The 2006 (Association) Football World Cup](https://doi.org/10.1057/jors.2009.127). *Journal of the Operational Research Society*, 61(10), 1530-1539.** The predictive parameters below correspond to equations 12 and 14, and the match-outcome aggregation to equations 15-17.

For a team with historical goal counts $x_i$ against opponents rated $R_i$, and a new opponent rated $R_*$, the core calculation is:

$$
r = \frac{1}{2} + \sum_i x_i + w s,
\qquad
\beta = \sum_i \frac{1}{R_i} + \frac{w}{R_*},
\qquad
p = \frac{\beta}{\beta + 1/R_*}.
$$

The predicted goal count follows:

$$
X_* \sim \operatorname{NegativeBinomial}(r,p).
$$

Setting $\theta=R_{\mathrm{team}}\lambda$ in the paper gives $X_i \mid \theta \sim \operatorname{Poisson}(\theta/R_i)$ and a Gamma posterior for $\theta$ with shape $r$ and rate $\beta$. The half-unit shape offset comes from the Jeffreys baseline. The optional power-prior term treats $s$ as a suggested goal count against the current opponent, weighted by $w$. Here $w$ is the aggregate expert weight ($a_0$ times the number of experts in the paper), and $s$ is their mean suggestion. The original default, retained here, is **$w=0$**, so the explicit expert suggestion is inactive. A positive rate from historical matches is then required for a proper posterior.

For example, `--prior-weight 1 --prior-goals 1.5` adds one weighted pseudo-match for each team. This small demonstration uses the same suggestion for both sides, simplifying the paper's separate expert predictions. It covers match-level prediction rather than the paper's full World Cup simulation and experiments.

Historical games contribute equally. The mean goal prediction is $r(1-p)/p$. Multiplying every rating by the same positive constant leaves the predictions unchanged. Arbitrary rating *scales*, such as raw Elo points versus a transformed strength ratio, are not otherwise interchangeable.

SciPy's [`nbinom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html) supplies the probability mass function and tail calculations; its [`gamma`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html) documentation describes the shape/rate convention used in the mixture interpretation.

## Numerical checks and limits

Each goal distribution is truncated using an inverse survival function, with its remaining tail checked explicitly. The default combined omitted probability is at most `1e-10`. Probabilities are **not renormalized**, and missing tail mass is not assigned to the draw outcome:

$$
P(\text{home win})+P(\text{draw})+P(\text{away win})
+P(\text{omitted}) \approx 1.
$$

A distribution requiring more than 200 goals of support raises an error instead of silently clipping it. The tests check the original equations, the Gamma-Poisson mixture, probability accounting, symmetry, rating-scale invariance, temporal input rules, malformed inputs and the complete command-line example.

```bash
python -m unittest discover -s tests -v
```

This is an exploratory model. It assumes independent team goal counts, has no separately estimated home advantage or attack/defense parameters, and applies one supplied rating per team to the selected history. A dated snapshot does not reconstruct the ratings available at each historical match. The bundled example and numerical tests do not establish out-of-sample accuracy, calibration, or betting returns.
