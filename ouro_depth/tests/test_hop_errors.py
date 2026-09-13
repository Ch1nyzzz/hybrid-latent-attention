import unittest

from ouro_depth.analyze_hop_errors import cycle_distances, target_stats, transitions


class HopErrorsTests(unittest.TestCase):
    def test_cycle_distances_and_disconnected_cycle_rejected(self):
        edges = [(str(i), str((i + 1) % 25)) for i in range(25)]
        distances = cycle_distances(list(reversed(edges)), "3")
        self.assertEqual(distances["3"], 0)
        self.assertEqual(distances["2"], 24)
        self.assertEqual(distances["9"], 6)
        with self.assertRaises(ValueError):
            cycle_distances([("a", "b"), ("b", "a"), ("c", "c")], "a")

    def test_availability_conditioned_denominators(self):
        rows = [
            {"requested_hop": 6, "choice_hop": 4, "offered_hops": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"requested_hop": 6, "choice_hop": 6, "offered_hops": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"requested_hop": 6, "choice_hop": 9, "offered_hops": [0, 1, 2, 3, 5, 6, 7, 9]},
        ]
        stat = target_stats(rows, [4])
        self.assertEqual((stat["eligible_rows"], stat["selected_rows"]), (2, 1))
        self.assertEqual(stat["random_expected_selected"], 2 / 8)
        wrong = target_stats(rows, [4], True)
        self.assertEqual((wrong["eligible_rows"], wrong["selected_rows"]), (1, 1))
        self.assertEqual(wrong["random_expected_selected"], 1 / 7)
        region = target_stats(rows, [1, 2, 3, 4], True)
        self.assertEqual(region["random_expected_selected"], 4 / 7 + 3 / 7)
        self.assertEqual(target_stats(rows, [6], True)["eligible_rows"], 0)

    def test_paired_correctness_and_positions(self):
        base = {"requested_hop": 3, "offered_hops": list(range(8))}
        before = [{**base, "id": "a", "choice_hop": 3, "raw_hop": 3}, {**base, "id": "b", "choice_hop": 2, "raw_hop": 2}]
        after = [{**base, "id": "a", "choice_hop": 4, "raw_hop": None}, {**base, "id": "b", "choice_hop": 3, "raw_hop": 3}]
        got = transitions(before, after)
        self.assertEqual(got["choice_correctness"], {"right_to_wrong": 1, "wrong_to_right": 1})
        self.assertEqual(got["newly_wrong_hop_counts"][4], 1)
        self.assertEqual(got["corrected_previous_hop_counts"][2], 1)
        with self.assertRaises(ValueError):
            transitions(before, list(reversed(after)))


if __name__ == "__main__":
    unittest.main()
