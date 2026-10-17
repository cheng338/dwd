"""Fit-free checks of the standalone finite-grid reporting helper."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import _grid_report as grid_report


def candidate(lambd=1.0, gamma=1.0, **other):
    return {"lambd": lambd, "kernel_kws": {"gamma": gamma}, **other}


def folds(candidates, correct, counts=None):
    """Use actual resumable-CV field shapes; rows index folds."""
    if counts is None:
        counts = [10] * len(correct)
    result = []
    for successes, size in zip(correct, counts):
        values = [value / size for value in successes]
        result.append({
            "params": deepcopy(candidates),
            "test_correct": list(successes),
            "test_count": [size] * len(candidates),
            "test_score": values,
            "train_correct": list(successes),
            "train_count": [size] * len(candidates),
            "train_score": values.copy(),
            "runtime": [0.0] * len(candidates),
            "init_time": 0.0,
        })
    return result


def exact_best(candidates, records):
    means = [
        sum((Fraction(row["test_correct"][i], row["test_count"][i])
             for row in records), Fraction()) / len(records)
        for i in range(len(candidates))
    ]
    return deepcopy(candidates[max(range(len(candidates)), key=means.__getitem__)])


class GridReportTests(unittest.TestCase):
    def report(self, candidates, records, best=None):
        if best is None:
            best = exact_best(candidates, records)
        original = deepcopy((candidates, records, best))
        with patch("dwd.gen_kern_dwd.KernGDWD.fit",
                   side_effect=AssertionError("report must not fit a classifier")):
            report = grid_report.summarize_grid(candidates, records, best)
        self.assertEqual((candidates, records, best), original)
        self.assertEqual(report["status"], "available")
        self.assertEqual(report["scope"], "best_among_supplied_candidates")
        json.dumps(report, allow_nan=False)
        return report

    def axis(self, report, path):
        matching = [axis for axis in report["selected_axes"] if axis["path"] == path]
        self.assertEqual(len(matching), 1, path)
        return matching[0]

    def check_fraction(self, report, key, expected):
        self.assertEqual(report[key + "_exact"],
                         {"numerator": expected.numerator,
                          "denominator": expected.denominator})
        self.assertEqual(report[key], float(expected))

    def assert_unavailable(self, candidates, records, best):
        original = deepcopy((candidates, records, best))
        result = grid_report.safe_summarize_grid(candidates, records, best)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["scope"], "best_among_supplied_candidates")
        self.assertIsInstance(result["reporting_error"], str)
        self.assertEqual((candidates, records, best), original)
        json.dumps(result, allow_nan=False)

    def test_interior_winner_has_conditional_brackets(self):
        candidates = [candidate(value, gamma)
                      for value in (1.0, 2.0, 4.0) for gamma in (.1, 1.0, 10.0)]
        records = folds(candidates, [[1, 2, 3, 2, 9, 4, 1, 5, 3]] * 2)
        report = self.report(candidates, records)
        self.assertEqual(report["best_index"], 4)
        for path in ("lambd", "kernel_kws.gamma"):
            axis = self.axis(report, path)
            self.assertEqual(axis["status"], "known")
            self.assertEqual(axis["position"], "interior")
            self.assertEqual(axis["slice_candidate_count"], 3)
            self.assertEqual(axis["distinct_numeric_value_count"], 3)

    def test_lower_and_upper_endpoints_ignore_input_sort_order(self):
        candidates = [candidate(value) for value in (4.0, 1.0, 2.0)]
        for scores, expected_index, position in (
                ([1, 9, 2], 1, "lower"), ([9, 1, 2], 0, "upper")):
            with self.subTest(position=position):
                report = self.report(candidates, folds(candidates, [scores] * 2))
                self.assertEqual(report["best_index"], expected_index)
                axis = self.axis(report, "lambd")
                self.assertEqual(axis["position"], position)
                self.assertEqual((axis["minimum"], axis["maximum"]), (1.0, 4.0))

    def test_conditional_slice_does_not_use_global_extrema(self):
        candidates = [candidate(value, .1) for value in (1.0, 2.0, 4.0)]
        candidates.append(candidate(.1, 10.0))
        records = folds(candidates, [[9, 7, 6, 5]] * 2)
        report = self.report(candidates, records)
        axis = self.axis(report, "lambd")
        self.assertEqual(axis["position"], "lower")
        self.assertEqual(axis["minimum"], 1.0)
        self.assertEqual(axis["slice_candidate_count"], 3)
        self.assertEqual(self.axis(report, "kernel_kws.gamma")["position"], "singleton")

    def test_other_kernel_keywords_remain_fixed_in_slice(self):
        candidates = [candidate(2.0, 1.0), candidate(2.0, 2.0), candidate(2.0, .1)]
        candidates[0]["kernel_kws"]["coef0"] = 0.0
        candidates[1]["kernel_kws"]["coef0"] = 0.0
        candidates[2]["kernel_kws"]["coef0"] = 1.0
        report = self.report(candidates, folds(candidates, [[9, 7, 5]] * 2))
        axis = self.axis(report, "kernel_kws.gamma")
        self.assertEqual(axis["position"], "lower")
        self.assertEqual(axis["minimum"], 1.0)
        self.assertEqual(axis["slice_candidate_count"], 2)

    def test_exact_fraction_tie_ignores_float_average_rounding(self):
        candidates = [candidate(.2), candidate(.08)]
        records = folds(candidates, [[0, 1], [1, 2], [5, 3]])
        report = self.report(candidates, records)
        self.assertEqual(report["best_index"], 0)
        self.assertEqual(report["best_params"], candidates[0])
        self.assertEqual(report["tied_best_indices"], [0, 1])
        self.assertEqual(report["tied_best_count"], 2)
        self.check_fraction(report, "best_score", Fraction(1, 5))
        self.assertEqual(report["gap_to_second_candidate"], 0.0)
        self.assertIsNone(report["gap_to_next_strictly_lower"])

    def test_unequal_fold_sizes_retain_equal_fold_weight(self):
        candidates = [candidate(1.0), candidate(2.0)]
        records = folds(candidates, [[4, 0], [0, 6]], counts=[4, 8])
        report = self.report(candidates, records)
        self.assertEqual(report["best_index"], 0)
        self.check_fraction(report, "best_score", Fraction(1, 2))
        self.check_fraction(report, "gap_to_second_candidate", Fraction(1, 8))

    def test_one_correct_prediction_difference_is_not_a_tie(self):
        candidates = [candidate(1.0), candidate(2.0)]
        report = self.report(candidates, folds(candidates, [[0, 0], [1, 1], [5, 6]]))
        self.assertEqual(report["best_index"], 1)
        self.assertEqual(report["tied_best_count"], 1)
        self.check_fraction(report, "gap_to_second_candidate", Fraction(1, 30))

    def test_tied_top_and_next_lower_gaps_are_distinct(self):
        candidates = [candidate(value) for value in (1.0, 2.0, 3.0)]
        report = self.report(candidates, folds(candidates, [[8, 8, 7]] * 2))
        self.assertEqual(report["tied_best_count"], 2)
        self.check_fraction(report, "gap_to_second_candidate", Fraction(0))
        self.check_fraction(report, "gap_to_next_strictly_lower", Fraction(1, 10))
        self.assertFalse(report["all_flat"])

    def test_duplicate_settings_are_preserved_as_distinct_candidates(self):
        candidates = [candidate(1.0), candidate(1.0), candidate(2.0)]
        report = self.report(candidates, folds(candidates, [[8, 8, 7]] * 2))
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(report["tied_best_count"], 2)
        self.assertEqual(report["tied_best_indices"], [0, 1])
        self.assertEqual(report["best_index"], 0)
        self.assertEqual(self.axis(report, "lambd")["slice_candidate_count"], 3)
        self.assertEqual(self.axis(report, "lambd")["distinct_numeric_value_count"], 2)

    def test_adjacent_float_parameters_are_not_merged_or_rounded(self):
        mid = math.nextafter(1.0, math.inf)
        upper = math.nextafter(mid, math.inf)
        candidates = [candidate(value) for value in (1.0, mid, upper)]
        report = self.report(candidates, folds(candidates, [[7, 9, 6]] * 2))
        axis = self.axis(report, "lambd")
        self.assertEqual(report["best_params"]["lambd"], mid)
        self.assertEqual(axis["value"], mid)
        self.assertEqual(axis["distinct_numeric_value_count"], 3)
        self.assertEqual(axis["position"], "interior")
        self.assertEqual(axis["maximum"], upper)

    def test_singleton_has_no_score_separation_or_endpoint_claim(self):
        candidates = [candidate()]
        report = self.report(candidates, folds(candidates, [[8], [9]]))
        self.assertTrue(report["singleton"])
        self.assertFalse(report["all_flat"])
        self.assertIsNone(report["gap_to_second_candidate"])
        self.assertIsNone(report["gap_to_next_strictly_lower"])
        self.assertEqual(self.axis(report, "lambd")["position"], "singleton")

    def test_all_flat_tied_details_are_bounded(self):
        candidates = [candidate(float(index + 1)) for index in range(37)]
        report = self.report(candidates, folds(candidates, [[5] * 37] * 2))
        self.assertTrue(report["all_flat"])
        self.assertFalse(report["singleton"])
        self.assertEqual(report["tied_best_count"], 37)
        self.assertEqual(report["tied_best_indices"], list(range(10)))
        self.assertEqual(len(report["tied_best_details"]), 10)
        self.assertEqual([item["candidate_index"] for item in report["tied_best_details"]],
                         list(range(10)))
        self.assertTrue(report["tied_best_details_truncated"])
        self.assertIsNone(report["gap_to_next_strictly_lower"])

    def test_unsupported_axis_is_explicitly_unknown(self):
        for other_gamma in ("scale", 1.0):
            with self.subTest(other_gamma=other_gamma):
                candidates = [candidate(1.0, "scale"), candidate(1.0, other_gamma)]
                report = self.report(candidates, folds(candidates, [[8, 7]] * 2))
                axis = self.axis(report, "kernel_kws.gamma")
                self.assertEqual(axis["status"], "unknown")
                self.assertIsNone(axis["position"])
                self.assertTrue(axis["reason"])

    def test_missing_selected_axis_is_explicitly_unknown(self):
        candidates = [{"lambd": 1.0, "kernel_kws": {}}, candidate(2.0, 1.0)]
        report = self.report(candidates, folds(candidates, [[8, 7]] * 2))
        axis = self.axis(report, "kernel_kws.gamma")
        self.assertEqual(axis["status"], "unknown")
        self.assertIsNone(axis["position"])
        self.assertTrue(axis["reason"])

    def test_malformed_or_incomplete_counts_report_unavailable(self):
        candidates = [candidate(1.0), candidate(2.0)]
        base = folds(candidates, [[8, 7]] * 2)
        mutations = [
            lambda rows: rows[0]["test_correct"].pop(),
            lambda rows: rows[0]["test_count"].append(10),
            lambda rows: rows[0].pop("test_correct"),
            lambda rows: rows[0]["test_count"].__setitem__(0, 0),
            lambda rows: rows[0]["test_count"].__setitem__(0, True),
            lambda rows: rows[0]["test_correct"].__setitem__(0, 11),
            lambda rows: rows[0]["test_correct"].__setitem__(0, 8.0),
            lambda rows: rows[0]["test_score"].__setitem__(0, .7),
            lambda rows: rows[0]["test_score"].pop(),
            lambda rows: rows[0]["params"].reverse(),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                records = deepcopy(base)
                mutate(records)
                self.assert_unavailable(candidates, records, candidates[0])
        self.assert_unavailable(candidates, [], candidates[0])

    def test_wrong_supplied_winner_is_not_silently_reselected(self):
        candidates = [candidate(1.0), candidate(2.0)]
        records = folds(candidates, [[8, 7]] * 2)
        self.assert_unavailable(candidates, records, candidates[1])

    def test_safe_wrapper_failure_does_not_expose_exception_message(self):
        candidates = [candidate()]
        records = folds(candidates, [[8], [9]])
        with patch.object(grid_report, "summarize_grid",
                          side_effect=RuntimeError("private diagnostic sentinel")):
            result = grid_report.safe_summarize_grid(candidates, records, candidates[0])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reporting_error"], "RuntimeError")
        self.assertNotIn("private diagnostic sentinel", json.dumps(result))

    def test_replayed_identical_records_produce_same_report(self):
        candidates = [candidate(1.0), candidate(2.0), candidate(4.0)]
        records = folds(candidates, [[7, 9, 8], [6, 9, 7]])
        first = self.report(candidates, records)
        replay = self.report(deepcopy(candidates), deepcopy(records))
        self.assertEqual(first, replay)

    def test_numpy_fixed_parameter_type_does_not_merge_slices(self):
        candidates = [candidate(1.0, q=np.int64(1)), candidate(2.0, q=1)]
        report = self.report(candidates, folds(candidates, [[8, 7]] * 2))
        axis = self.axis(report, "lambd")
        self.assertEqual(axis["slice_candidate_count"], 1)
        self.assertEqual(axis["position"], "singleton")

    def test_numpy_label_type_mismatch_is_not_verified(self):
        candidates = [candidate(1.0, q=np.int64(1)), candidate(2.0, q=1)]
        records = folds(candidates, [[8, 7]] * 2)
        changed_best = deepcopy(candidates[0])
        changed_best["q"] = 1
        self.assert_unavailable(candidates, records, changed_best)
        changed_records = deepcopy(records)
        changed_records[0]["params"][0]["q"] = 1
        self.assert_unavailable(candidates, changed_records, candidates[0])

    def test_formatter_qualifies_finite_grid_and_singleton(self):
        candidates = [candidate()]
        report = self.report(candidates, folds(candidates, [[8], [9]]))
        before = deepcopy(report)
        text = grid_report.format_grid_report(report)
        self.assertIn("Best among 1 supplied candidates", text)
        self.assertIn("Only one candidate", text)
        self.assertIn("no endpoint evidence", text)
        self.assertIn("does not establish an optimum beyond", text)
        self.assertEqual(report, before)

    def test_formatter_bounds_details_and_escapes_control_characters(self):
        unusual = "gamma\nFORGED-LINE\r\t\x1b" + "x" * 1000
        params = {f"p{index:02}": 1.0 for index in range(25)}
        params[unusual] = 2.0
        candidates = [params]
        report = self.report(candidates, folds(candidates, [[8], [9]]))
        self.assertEqual(report["selected_axis_count"], 26)
        self.assertEqual(len(report["selected_axes"]), 10)
        self.assertTrue(report["selected_axes_truncated"])
        text = grid_report.format_grid_report(report)
        self.assertEqual(sum(line.startswith("Coordinate ") and not line.startswith(
            "Coordinate details") for line in text.splitlines()), 10)
        self.assertLessEqual(len(text.splitlines()), 22)
        self.assertLessEqual(max(map(len, text.splitlines())), 300)
        self.assertFalse(any(line.startswith("FORGED-LINE") for line in text.splitlines()))
        for character in ("\r", "\t", "\x1b"):
            self.assertNotIn(character, text)

    def test_formatter_failure_is_safe_and_has_no_exception_detail(self):
        for malformed in (None, [], {}, {"status": "available"},
                          {"status": "unavailable", "reporting_error": "private sentinel"}):
            with self.subTest(value=malformed):
                text = grid_report.format_grid_report(malformed)
                self.assertEqual(text, "Finite-grid report unavailable; the selected result is unchanged.")
                self.assertNotIn("private sentinel", text)


if __name__ == "__main__":
    unittest.main()

