"""Numerical, data-boundary, and end-to-end checks for the standalone sample.

The parameter references below are direct scalar translations of the earlier
Europred equations. The mixture check independently integrates a Gamma-Poisson
model; neither check measures predictive accuracy on football matches.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import gamma, poisson


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import europred  # noqa: E402


class EuropredTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.history = pd.DataFrame(
            [
                ["2024-04-01", "A", "C", 2, 0],
                ["2024-04-02", "B", "D", 1, 1],
                ["2024-04-10", "C", "A", 1, 0],
                ["2024-04-11", "D", "B", 0, 2],
            ],
            columns=["Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore"],
        )
        self.ratings = pd.DataFrame(
            [[name, value, "2024-03-01"] for name, value in
             [("A", 1500), ("B", 1400), ("C", 1300), ("D", 1200)]],
            columns=["Name", "Rating", "AsOf"],
        )
        self.fixtures = pd.DataFrame(
            [["2024-06-01", "A", "B"]],
            columns=["Date", "HomeTeam", "AwayTeam"],
        )

    def write_inputs(self, history=None, ratings=None, fixtures=None):
        paths = tuple(self.root / name for name in
                      ("history.csv", "ratings.csv", "fixtures.csv"))
        frames = (
            self.history if history is None else history,
            self.ratings if ratings is None else ratings,
            self.fixtures if fixtures is None else fixtures,
        )
        for frame, path in zip(frames, paths):
            frame.to_csv(path, index=False)
        return paths

    def inputs(self, history=None, ratings=None, fixtures=None):
        return europred.load_inputs(*self.write_inputs(history, ratings, fixtures))

    def test_parameters_match_original_equations_with_and_without_prior(self):
        goals = [1, 2, 0, 3]
        opponents = [1000, 1200, 900, 1500]
        target_rating = 1100
        for prior_weight, prior_goals in [(0.0, 1.0), (1.75, 1.4)]:
            with self.subTest(prior_weight=prior_weight):
                model = europred.fit_goal_model(
                    goals, opponents, target_rating,
                    prior_weight=prior_weight, prior_goals=prior_goals,
                )
                expected_shape = sum(goals) + prior_weight * prior_goals + 0.5
                expected_rate = (sum(1.0 / value for value in opponents)
                                 + prior_weight / target_rating)
                expected_p = expected_rate / (expected_rate + 1.0 / target_rating)
                self.assertAlmostEqual(model.shape, expected_shape, places=13)
                self.assertAlmostEqual(model.rate, expected_rate, places=15)
                self.assertAlmostEqual(model.probability, expected_p, places=14)

    def test_predictive_pmf_matches_independent_gamma_poisson_integral(self):
        model = europred.fit_goal_model([1, 2, 0, 3], [1000, 1200, 900, 1500], 1100)
        pmf, _ = europred.goal_distribution(model, tail_tolerance=1e-12)
        # Integrate over t=lambda/current_opponent_rating so quadrature remains
        # well scaled even though the original team ratings are around 1000.
        rate_for_t = model.rate * 1100
        for goals in [0, 1, 3, 7]:
            with self.subTest(goals=goals):
                expected, error = quad(
                    lambda t: poisson.pmf(goals, t)
                    * gamma.pdf(t, a=model.shape, scale=1.0 / rate_for_t),
                    0.0, np.inf, epsabs=1e-12, epsrel=1e-10,
                )
                self.assertLess(error, 1e-9)
                self.assertAlmostEqual(pmf[goals], expected, places=10)

    def test_distribution_preserves_mass_and_respects_tail_tolerance(self):
        model = europred.fit_goal_model([0, 1, 2], [1300, 1450, 1500], 1400)
        tolerance = 1e-10
        pmf, omitted = europred.goal_distribution(model, tail_tolerance=tolerance)
        self.assertEqual(pmf.ndim, 1)
        self.assertGreater(len(pmf), 1)
        self.assertTrue(np.all(np.isfinite(pmf)))
        self.assertTrue(np.all(pmf >= 0))
        self.assertGreaterEqual(omitted, 0)
        self.assertLessEqual(omitted, tolerance * (1 + 1e-5))
        self.assertAlmostEqual(float(pmf.sum()) + omitted, 1.0, places=12)

    def test_uniform_rating_rescaling_leaves_predictions_unchanged(self):
        original = europred.fit_goal_model([1, 0, 3], [1100, 1600, 1400], 1300,
                                           prior_weight=0.75, prior_goals=1.3)
        scaled = europred.fit_goal_model([1, 0, 3], [11000, 16000, 14000], 13000,
                                         prior_weight=0.75, prior_goals=1.3)
        self.assertAlmostEqual(original.probability, scaled.probability, places=14)
        left, _ = europred.goal_distribution(original)
        right, _ = europred.goal_distribution(scaled)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-14)

    def test_joint_matrix_symmetry_and_team_swap(self):
        history, ratings, fixtures = self.inputs()
        fixture = fixtures.iloc[0]
        summary, joint = europred.predict_fixture(history, ratings, fixture)
        swapped = fixture.copy()
        swapped["HomeTeam"], swapped["AwayTeam"] = fixture["AwayTeam"], fixture["HomeTeam"]
        reversed_summary, reversed_joint = europred.predict_fixture(history, ratings, swapped)
        np.testing.assert_allclose(joint, reversed_joint.T, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(summary["HomeWin"], reversed_summary["AwayWin"], places=12)
        self.assertAlmostEqual(summary["Draw"], reversed_summary["Draw"], places=12)
        self.assertAlmostEqual(summary["HomeWin"], float(np.tril(joint, -1).sum()), places=12)
        self.assertAlmostEqual(summary["Draw"], float(np.trace(joint)), places=12)
        self.assertAlmostEqual(summary["AwayWin"], float(np.triu(joint, 1).sum()), places=12)
        self.assertAlmostEqual(float(joint.sum()) + summary["OmittedProbability"], 1.0, places=12)

        symmetric_history = self.history.copy()
        symmetric_history.loc[1, "HomeScore"] = 2
        symmetric_history.loc[3, "AwayScore"] = 0
        symmetric_ratings = self.ratings.copy()
        symmetric_ratings["Rating"] = 1500
        h, r, f = self.inputs(symmetric_history, symmetric_ratings)
        symmetric, matrix = europred.predict_fixture(h, r, f.iloc[0])
        self.assertAlmostEqual(symmetric["HomeWin"], symmetric["AwayWin"], places=12)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)

    def test_same_day_and_future_history_are_excluded(self):
        history, ratings, fixtures = self.inputs()
        before, joint_before = europred.predict_fixture(history, ratings, fixtures.iloc[0])
        future = pd.DataFrame(
            [["2024-06-01", "A", "B", 15, 12],
             ["2024-07-01", "A", "B", 20, 18]], columns=self.history.columns,
        )
        h, r, f = self.inputs(pd.concat([self.history, future], ignore_index=True))
        after, joint_after = europred.predict_fixture(h, r, f.iloc[0])
        for key in ["HomeWin", "Draw", "AwayWin", "HomeExpectedGoals", "AwayExpectedGoals",
                    "HomeHistoryMatches", "AwayHistoryMatches"]:
            self.assertEqual(before[key], after[key])
        np.testing.assert_array_equal(joint_before, joint_after)
        self.assertEqual(after["HomeHistoryMatches"], 2)
        self.assertEqual(after["AwayHistoryMatches"], 2)

    def test_rating_snapshot_after_fixture_is_rejected(self):
        future_ratings = self.ratings.copy()
        future_ratings.loc[0, "AsOf"] = "2024-06-02"
        with self.assertRaises(ValueError):
            history, ratings, fixtures = self.inputs(ratings=future_ratings)
            europred.predict_fixture(history, ratings, fixtures.iloc[0])

    def test_missing_columns_duplicate_names_and_unrated_teams_are_rejected(self):
        self_match_history = self.history.copy()
        self_match_history.loc[0, "AwayTeam"] = self_match_history.loc[0, "HomeTeam"]
        self_match_fixture = self.fixtures.copy()
        self_match_fixture.loc[0, "AwayTeam"] = self_match_fixture.loc[0, "HomeTeam"]
        cases = [
            (self.history.drop(columns="HomeScore"), self.ratings, self.fixtures),
            (self.history, pd.concat([self.ratings, self.ratings.iloc[[0]]], ignore_index=True), self.fixtures),
            (self.history, self.ratings[self.ratings["Name"] != "C"], self.fixtures),
            (pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True), self.ratings, self.fixtures),
            (self_match_history, self.ratings, self.fixtures),
            (self.history, self.ratings, self_match_fixture),
        ]
        for history, ratings, fixtures in cases:
            with self.subTest(history_columns=list(history.columns), rating_names=list(ratings["Name"])):
                with self.assertRaises(ValueError):
                    h, r, f = self.inputs(history, ratings, fixtures)
                    europred.predict_fixture(h, r, f.iloc[0])

    def test_invalid_scores_ratings_dates_and_model_inputs_are_rejected(self):
        for bad_score in [-1, 1.5, np.nan]:
            with self.subTest(score=bad_score):
                history = self.history.astype({"HomeScore": float}).copy()
                history.loc[0, "HomeScore"] = bad_score
                with self.assertRaises(ValueError):
                    self.inputs(history=history)
        for bad_rating in [0, -100, np.inf]:
            with self.subTest(rating=bad_rating):
                ratings = self.ratings.astype({"Rating": float}).copy()
                ratings.loc[0, "Rating"] = bad_rating
                with self.assertRaises(ValueError):
                    self.inputs(ratings=ratings)
        history = self.history.copy()
        history.loc[0, "Date"] = "not-a-date"
        with self.assertRaises(ValueError):
            self.inputs(history=history)
        for goals, ratings, opponent in [([1], [0], 1000), ([1, 2], [1000], 1000),
                                          ([1], [1000], 0), ([-1], [1000], 1000)]:
            with self.subTest(model_inputs=(goals, ratings, opponent)):
                with self.assertRaises(ValueError):
                    europred.fit_goal_model(goals, ratings, opponent)

    def test_no_history_requires_explicit_prior(self):
        with self.assertRaises(ValueError):
            europred.fit_goal_model([], [], 1500)
        model = europred.fit_goal_model([], [], 1500, prior_weight=1.0, prior_goals=1.0)
        self.assertAlmostEqual(model.shape, 1.5)
        self.assertAlmostEqual(model.rate, 1.0 / 1500)
        self.assertAlmostEqual(model.probability, 0.5)
        no_target_history = self.history[
            ~self.history["HomeTeam"].isin(["A", "B"])
            & ~self.history["AwayTeam"].isin(["A", "B"])
        ]
        h, r, f = self.inputs(history=no_target_history)
        with self.assertRaises(ValueError):
            europred.predict_fixture(h, r, f.iloc[0])
        result, _ = europred.predict_fixture(h, r, f.iloc[0], prior_weight=1.0)
        self.assertEqual(result["HomeHistoryMatches"], 0)
        self.assertEqual(result["AwayHistoryMatches"], 0)

    def test_extreme_support_is_rejected_instead_of_silently_truncated(self):
        model = europred.GoalModel(shape=1000.0, rate=1.0, probability=1e-6)
        with self.assertRaises(ValueError):
            europred.goal_distribution(model, max_goals=200)

    def test_tail_tolerance_boundaries_and_invalid_values(self):
        model = europred.fit_goal_model([0, 1, 2], [1300, 1450, 1500], 1400)
        for tolerance in [1e-15, 1e-2]:
            with self.subTest(valid_tolerance=tolerance):
                pmf, omitted = europred.goal_distribution(model, tail_tolerance=tolerance)
                self.assertLessEqual(omitted, tolerance * (1 + 1e-5))
                self.assertAlmostEqual(float(pmf.sum()) + omitted, 1.0, places=12)
        for tolerance in [0.0, -1e-10, 1.0, np.nan, np.inf]:
            with self.subTest(invalid_tolerance=tolerance):
                with self.assertRaises(ValueError):
                    europred.goal_distribution(model, tail_tolerance=tolerance)

    def test_run_writes_readable_predictions_and_complete_score_matrix(self):
        output = self.root / "output"
        result = europred.run(*self.write_inputs(), output)
        saved = pd.read_csv(output / "predictions.csv")
        matrix = pd.read_csv(output / "score_matrix_001.csv", index_col=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "HomeTeam"], "A")
        self.assertEqual(saved.loc[0, "AwayTeam"], "B")
        self.assertGreater(matrix.shape[0], 1)
        self.assertGreater(matrix.shape[1], 1)
        self.assertTrue(np.isfinite(matrix.to_numpy(dtype=float)).all())
        self.assertAlmostEqual(matrix.to_numpy(dtype=float).sum()
                               + saved.loc[0, "OmittedProbability"], 1.0, places=11)

    def test_reused_output_is_rejected_without_changing_previous_results(self):
        output = self.root / "output"
        paths = self.write_inputs()
        europred.run(*paths, output)
        previous = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "fresh output directory"):
            europred.run(*paths, output)
        self.assertEqual(previous, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_cli_runs_from_an_unrelated_working_directory(self):
        history, ratings, fixtures = self.write_inputs()
        output = self.root / "cli_output"
        other_cwd = self.root / "unrelated"
        other_cwd.mkdir()
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "europred.py"),
             "--history", str(history), "--ratings", str(ratings),
             "--fixtures", str(fixtures), "--output", str(output)],
            cwd=other_cwd, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((output / "predictions.csv").is_file())
        self.assertTrue((output / "score_matrix_001.csv").is_file())
        self.assertEqual(list(other_cwd.iterdir()), [])

    def test_import_has_no_printing_or_file_creation(self):
        other_cwd = self.root / "import_check"
        other_cwd.mkdir()
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            "sys.dont_write_bytecode = True; import europred"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=other_cwd,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(list(other_cwd.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
