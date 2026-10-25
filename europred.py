"""Europred: historical football results to Bayesian score probabilities.

Standalone adaptation of my earlier Europred project for code review. Local
machine paths have been replaced with explicit CSV inputs and a synthetic demo.
Statistical model: Suzuki et al. (2010), doi:10.1057/jors.2009.127.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom


MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_ROWS = 25_000
MAX_FIXTURES = 50
MAX_GOALS = 200
DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class GoalModel:
    """Gamma posterior (shape, rate) and its negative-binomial probability."""

    shape: float
    rate: float
    probability: float


def fit_goal_model(
    goals,
    opponent_ratings,
    current_opponent_rating: float,
    *,
    prior_weight: float = 0.0,
    prior_goals: float = 1.0,
) -> GoalModel:
    """Fit the original Europred posterior-predictive parameters.

    In the rescaled Poisson model, a historical opponent contributes exposure
    1 / rating. The 0.5 shape offset comes from the Jeffreys baseline. Optional
    expert input is one weighted pseudo-match against the current opponent.
    """
    goals = np.asarray(goals, dtype=float)
    ratings = np.asarray(opponent_ratings, dtype=float)
    if goals.ndim != 1 or goals.shape != ratings.shape:
        raise ValueError("Goals and opponent ratings must be matching 1D arrays.")
    if (
        not np.all(np.isfinite(goals))
        or np.any(goals < 0)
        or np.any(goals != np.floor(goals))
    ):
        raise ValueError("Historical goals must be finite nonnegative integers.")
    if not np.all(np.isfinite(ratings)) or np.any(ratings <= 0):
        raise ValueError("Opponent ratings must be finite and positive.")
    if not np.isfinite(current_opponent_rating) or current_opponent_rating <= 0:
        raise ValueError("The current opponent rating must be finite and positive.")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("Prior weight must be finite and nonnegative.")
    if not np.isfinite(prior_goals) or prior_goals < 0:
        raise ValueError("Prior goals must be finite and nonnegative.")
    if not goals.size and prior_weight == 0:
        raise ValueError("No historical matches: supply history or a positive prior weight.")

    # Bayesian core: Suzuki et al. (2010), equations 12 and 14, as in Europred.
    shape = float(goals.sum() + prior_weight * prior_goals + 0.5)
    rate = float(np.sum(1.0 / ratings) + prior_weight / current_opponent_rating)
    probability = rate / (rate + 1.0 / current_opponent_rating)
    if not (
        np.isfinite(shape) and np.isfinite(rate)
        and rate > 0 and 0 < probability < 1
    ):
        raise ValueError("Inputs exceed the model's supported numerical range.")
    return GoalModel(shape, rate, probability)


def goal_distribution(
    model: GoalModel, *, tail_tolerance: float = 1e-10, max_goals: int = MAX_GOALS
) -> tuple[np.ndarray, float]:
    """Return an unrenormalized PMF and the explicitly omitted tail probability."""
    if not np.isfinite(tail_tolerance) or not 1e-15 <= tail_tolerance <= 1e-2:
        raise ValueError("Per-team tail tolerance must be between 1e-15 and 1e-2.")
    if not isinstance(max_goals, int) or not 1 <= max_goals <= MAX_GOALS:
        raise ValueError(f"Maximum goal support must be between 1 and {MAX_GOALS}.")
    if not (
        np.isfinite(model.shape) and model.shape > 0
        and 0 < model.probability < 1
    ):
        raise ValueError("Invalid negative-binomial parameters.")
    cutoff = nbinom.isf(tail_tolerance, model.shape, model.probability)
    if not np.isfinite(cutoff) or cutoff < 0 or cutoff > max_goals:
        raise ValueError("Distribution exceeds the supported score range; inspect inputs.")
    cutoff = int(cutoff)
    # Check the tail itself rather than relying on an individual small PMF value.
    while nbinom.sf(cutoff, model.shape, model.probability) > tail_tolerance:
        cutoff += 1
        if cutoff > max_goals:
            raise ValueError("Distribution exceeds the supported score range; inspect inputs.")
    scores = np.arange(cutoff + 1)
    probabilities = nbinom.pmf(scores, model.shape, model.probability)
    tail = float(nbinom.sf(cutoff, model.shape, model.probability))
    if not np.all(np.isfinite(probabilities)) or not np.isclose(
        probabilities.sum() + tail, 1, atol=1e-12, rtol=0
    ):
        raise ValueError("Could not compute a numerically consistent goal distribution.")
    return probabilities, tail


def _read_csv(
    path: Path, required_columns: set[str], *, allow_empty: bool = False
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Each input must be an existing CSV file no larger than 5 MiB.")
    frame = pd.read_csv(
        path, keep_default_na=False, compression=None, nrows=MAX_ROWS + 1
    )
    missing = required_columns - set(frame.columns)
    if missing or (frame.empty and not allow_empty) or len(frame) > MAX_ROWS:
        raise ValueError(
            f"Invalid input {path.name}: missing columns {sorted(missing)}, "
            "empty, or too many rows."
        )
    return frame


def _clean_names(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} contains an empty team name.")


def _parse_dates(frame: pd.DataFrame, column: str) -> None:
    text = frame[column].astype(str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"{column} must use YYYY-MM-DD dates.")
    frame[column] = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")


def load_inputs(
    history_path, ratings_path, fixtures_path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate each file once; keep model calculations free of file I/O."""
    history = _read_csv(
        history_path, {"Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore"},
        allow_empty=True,
    )
    ratings = _read_csv(ratings_path, {"Name", "Rating", "AsOf"})
    fixtures = _read_csv(fixtures_path, {"Date", "HomeTeam", "AwayTeam"})
    _clean_names(ratings, ("Name",))
    _parse_dates(ratings, "AsOf")
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="raise")
    if (
        not np.isfinite(ratings["Rating"]).all()
        or (ratings["Rating"] <= 0).any()
        or ratings["Name"].duplicated().any()
    ):
        raise ValueError("Ratings must be finite and positive, with one row per team.")
    ratings = ratings.set_index("Name")
    for frame in (history, fixtures):
        _clean_names(frame, ("HomeTeam", "AwayTeam"))
        _parse_dates(frame, "Date")
        if frame["HomeTeam"].eq(frame["AwayTeam"]).any():
            raise ValueError("A team cannot play itself.")
        if frame.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
            raise ValueError("Duplicate date/team fixtures would count a match twice.")
        teams = set(frame["HomeTeam"]) | set(frame["AwayTeam"])
        if teams - set(ratings.index):
            raise ValueError("Every historical and fixture team needs a rating.")
    for column in ("HomeScore", "AwayScore"):
        values = pd.to_numeric(history[column], errors="raise")
        if (
            not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError("Historical scores must be finite nonnegative integers.")
        history[column] = values.astype(float)
    if len(fixtures) > MAX_FIXTURES:
        raise ValueError(f"At most {MAX_FIXTURES} fixtures are supported per run.")
    return history, ratings, fixtures


def _team_history(
    history: pd.DataFrame, ratings: pd.DataFrame, team: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = history[
        (history["HomeTeam"] == team) | (history["AwayTeam"] == team)
    ]
    is_home = selected["HomeTeam"] == team
    goals = np.where(is_home, selected["HomeScore"], selected["AwayScore"])
    opponents = np.where(is_home, selected["AwayTeam"], selected["HomeTeam"])
    return goals, ratings.loc[opponents, "Rating"].to_numpy(dtype=float)


def predict_fixture(
    history: pd.DataFrame,
    ratings: pd.DataFrame,
    fixture: pd.Series,
    *,
    prior_weight: float = 0.0,
    prior_goals: float = 1.0,
    tail_tolerance: float = 1e-10,
) -> tuple[dict, np.ndarray]:
    """Predict from strictly earlier matches using one declared ratings snapshot."""
    if not np.isfinite(tail_tolerance) or not 1e-14 <= tail_tolerance <= 1e-2:
        raise ValueError("Total tail tolerance must be between 1e-14 and 1e-2.")
    date, home, away = fixture["Date"], fixture["HomeTeam"], fixture["AwayTeam"]
    if home == away or home not in ratings.index or away not in ratings.index:
        raise ValueError("A fixture needs two distinct rated teams.")
    if (ratings["AsOf"] >= date).any():
        raise ValueError("The ratings snapshot must predate the fixture date.")
    past = history[history["Date"] < date]
    models, distributions, tails, counts = [], [], [], []
    for team, opponent in ((home, away), (away, home)):
        goals, opponent_ratings = _team_history(past, ratings, team)
        model = fit_goal_model(
            goals, opponent_ratings, float(ratings.loc[opponent, "Rating"]),
            prior_weight=prior_weight, prior_goals=prior_goals,
        )
        distribution, tail = goal_distribution(model, tail_tolerance=tail_tolerance / 2)
        models.append(model)
        distributions.append(distribution)
        tails.append(tail)
        counts.append(len(goals))

    # Row i is home goals; column j is away goals. No rounding before summation.
    joint = np.outer(distributions[0], distributions[1])
    omitted = tails[0] + tails[1] - tails[0] * tails[1]
    summary = {
        "Date": date.strftime("%Y-%m-%d"), "HomeTeam": home, "AwayTeam": away,
        "HomeWin": float(np.tril(joint, k=-1).sum()),
        "Draw": float(np.trace(joint)),
        "AwayWin": float(np.triu(joint, k=1).sum()),
        "HomeExpectedGoals": float(
            nbinom.mean(models[0].shape, models[0].probability)
        ),
        "AwayExpectedGoals": float(
            nbinom.mean(models[1].shape, models[1].probability)
        ),
        "OmittedProbability": float(omitted),
        "HomeHistoryMatches": counts[0], "AwayHistoryMatches": counts[1],
    }
    return summary, joint


def run(
    history_path, ratings_path, fixtures_path, output_dir, **model_options
) -> pd.DataFrame:
    """Validate, predict all fixtures, then write summaries and full score grids."""
    output = Path(output_dir)
    if output.exists() and (
        not output.is_dir() or (output / "predictions.csv").exists()
        or any(output.glob("score_matrix_*.csv"))
    ):
        raise ValueError("Choose a fresh output directory to keep separate runs distinct.")
    history, ratings, fixtures = load_inputs(
        history_path, ratings_path, fixtures_path
    )
    results = [
        predict_fixture(history, ratings, row, **model_options)
        for _, row in fixtures.iterrows()
    ]
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([row for row, _ in results])
    summary.to_csv(output / "predictions.csv", index=False, float_format="%.15g")
    for i, (_, joint) in enumerate(results, start=1):
        matrix = pd.DataFrame(joint, columns=range(joint.shape[1]))
        matrix.index.name = "HomeGoals / AwayGoals"
        matrix.to_csv(output / f"score_matrix_{i:03d}.csv", float_format="%.15g")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA_DIR / "history.csv")
    parser.add_argument("--ratings", type=Path, default=DATA_DIR / "ratings.csv")
    parser.add_argument("--fixtures", type=Path, default=DATA_DIR / "fixtures.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-goals", type=float, default=1.0)
    parser.add_argument("--tail-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    try:
        summary = run(
            args.history, args.ratings, args.fixtures, args.output,
            prior_weight=args.prior_weight, prior_goals=args.prior_goals,
            tail_tolerance=args.tail_tolerance,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Input/model error: {error}\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote predictions.csv and {len(summary)} score matrices to {args.output}.")
    print("Bundled data are synthetic; running this demo does not measure predictive accuracy.")


if __name__ == "__main__":
    main()
