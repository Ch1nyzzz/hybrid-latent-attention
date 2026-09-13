"""Small stdlib fixtures for data/summary binding; no actual dataset access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ouro_depth.compare_predictions import _paired
from ouro_depth.v3_eval_binding import validate_evaluation, validate_initializer_launch


def paired_metrics(before, after):
    value = _paired(before, after)
    return {k: value[k] for k in ("gain", "wrong_to_right", "right_to_wrong", "bonferroni_wilson_approx_95ci", "mcnemar_exact_p")}


class EvaluationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v3-eval-binding-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.prefix, self.data_file = self.root / "eval", self.root / "dev.jsonl"
        def score(correct, choice, loss, choice_loss, mass, tied=False):
            return {"correct": correct, "choice_correct": correct, "choice": choice,
                    "nll": loss, "choice_nll": choice_loss, "answer_mass": mass,
                    "choice_tied": tied, "choice_tie_aware_correct": .5 if tied else float(correct)}
        self.predictions = [
            {"id": "one", "answer": "A", "family": "pointer_chasing", "difficulty": 1,
             "scores": {"4": score(True, "A", .5, .25, .8), "6": score(False, "B", 1., .75, .5), "8": score(True, "A", .5, .25, .8)}},
            {"id": "two", "answer": "B", "family": "pointer_chasing", "difficulty": 9,
             "scores": {"4": score(False, "A", 1.5, 1.25, .4), "6": score(True, "B", .25, .125, .9), "8": score(True, "B", .125, .0625, .95, True)}},
        ]
        self.data = [{k: v for k, v in row.items() if k != "scores"} for row in self.predictions]
        # Hand-calculated means; paired calculations reuse the already-tested
        # statistical primitive, independently specifying each expected vector.
        def metric(n, accuracy, nll, cnll, mass, ties, aware):
            return {"n": n, "accuracy": accuracy, "choice_accuracy": accuracy, "nll": nll,
                    "choice_nll": cnll, "answer_mass": mass, "choice_tie_rate": ties,
                    "choice_tie_aware_accuracy": aware}
        by_depth = [
            {"4": metric(2,.5,1.,.75,.6,0.,.5), "6": metric(2,.5,.625,.4375,.7,0.,.5), "8": metric(2,1.,.3125,.15625,.875,.5,.75)},
            {"4": metric(1,1.,.5,.25,.8,0.,1.), "6": metric(1,0.,1.,.75,.5,0.,0.), "8": metric(1,1.,.5,.25,.8,0.,1.)},
            {"4": metric(1,0.,1.5,1.25,.4,0.,0.), "6": metric(1,1.,.25,.125,.9,0.,1.), "8": metric(1,1.,.125,.0625,.95,1.,.5)},
        ]
        vectors = [{"4":[True,False],"6":[False,True],"8":[True,True]},
                   {"4":[True],"6":[False],"8":[True]}, {"4":[False],"6":[True],"8":[True]}]
        groups = []
        for index in range(3):
            answers = ["A","B"] if index == 0 else ["A" if index == 1 else "B"]
            groups.append({"by_depth":by_depth[index],
                "paired":{f"{a}->{b}/{field}":paired_metrics(vectors[index][a],vectors[index][b])
                          for a,b in [("4","8"),("4","6"),("6","8")] for field in ["correct","choice_correct"]},
                "answer_counts":{letter:answers.count(letter) for letter in "ABCDEFGH"},
                "majority_letter_baseline":.5 if index == 0 else 1.})
        self.summary = {"evaluator_version":2,"choice_tie_break":"ascending_token_id","depths":[4,6,8],"count":2,
            "metrics":{key:copy.deepcopy(groups[index]) for key,index in [
                ("all",0),("pointer_chasing",0),("pointer_chasing/d1",1),("easy",1),("pointer_chasing/d9",2),("hard",2)]}}
        self.write()

    def write(self):
        self.data_file.write_text("".join(json.dumps(row)+"\n" for row in self.data))
        Path(str(self.prefix)+".predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in self.predictions))
        Path(str(self.prefix)+".json").write_text(json.dumps(self.summary))

    def test_known_metrics_and_reordered_source_ids(self):
        self.data.reverse();self.write()
        result=validate_evaluation(self.prefix,self.data_file)
        self.assertEqual(result["count"],2)
        self.assertEqual(result["depths"],[4,6,8])
        self.assertTrue(result["all_score_derived_summary_metrics_match"])

    def test_wrong_source_id_or_metadata_is_rejected(self):
        for field,value in [("id","different"),("answer","C"),("difficulty",8),("family","other")]:
            original=copy.deepcopy(self.data)
            self.data[0][field]=value;self.write()
            with self.subTest(field=field),self.assertRaises(ValueError):validate_evaluation(self.prefix,self.data_file)
            self.data=original

    def test_binding_is_stable_and_identifies_prediction_changes_with_same_aggregates(self):
        original=validate_evaluation(self.prefix,self.data_file)
        self.assertEqual(original,validate_evaluation(self.prefix,self.data_file))
        self.predictions[0]['scores']['4']['prediction_token']=330
        self.write()
        changed=validate_evaluation(self.prefix,self.data_file)
        self.assertNotEqual(original['predictions_canonical_sha256'],changed['predictions_canonical_sha256'])
        self.assertEqual(original['summary_canonical_sha256'],changed['summary_canonical_sha256'])

    def test_stale_predictions_and_bad_t6_scores_are_rejected(self):
        self.predictions[0]["scores"]["6"]["nll"] += .1;self.write()
        with self.assertRaisesRegex(ValueError,"numeric mismatch"):validate_evaluation(self.prefix,self.data_file)
        self.predictions[0]["scores"]["6"]["nll"] = float("nan");self.write()
        with self.assertRaisesRegex(ValueError,"Nonfinite"):validate_evaluation(self.prefix,self.data_file)

    def test_summary_fields_groups_pairs_and_counts_are_checked(self):
        original=copy.deepcopy(self.summary)
        for field in ["n","accuracy","choice_accuracy","nll","choice_nll","answer_mass","choice_tie_rate","choice_tie_aware_accuracy"]:
            self.summary=copy.deepcopy(original)
            self.summary["metrics"]["hard"]["by_depth"]["6"][field] += 1
            self.write()
            with self.subTest(field=field),self.assertRaises(ValueError):validate_evaluation(self.prefix,self.data_file)
        for field in ["gain","wrong_to_right","right_to_wrong","bonferroni_wilson_approx_95ci","mcnemar_exact_p"]:
            self.summary=copy.deepcopy(original)
            pair=self.summary["metrics"]["all"]["paired"]["4->6/correct"]
            if isinstance(pair[field],list):pair[field][0] += .1
            else:pair[field] += 1
            self.write()
            with self.subTest(pair=field),self.assertRaises(ValueError):validate_evaluation(self.prefix,self.data_file)
        self.summary=copy.deepcopy(original);del self.summary["metrics"]["hard"];self.write()
        with self.assertRaisesRegex(ValueError,"key mismatch"):validate_evaluation(self.prefix,self.data_file)
        self.summary=copy.deepcopy(original);self.summary["metrics"]["all"]["answer_counts"]["A"]=2;self.write()
        with self.assertRaises(ValueError):validate_evaluation(self.prefix,self.data_file)

    def test_float_roundoff_allowed_but_integer_boolean_not(self):
        self.summary["metrics"]["all"]["by_depth"]["4"]["nll"] += 1e-12;self.write()
        validate_evaluation(self.prefix,self.data_file)
        self.summary["metrics"]["easy"]["by_depth"]["4"]["n"]=True;self.write()
        with self.assertRaisesRegex(ValueError,"integer mismatch"):validate_evaluation(self.prefix,self.data_file)

    def test_initializer_launch_is_bound_without_remote_files(self):
        remote=Path("/nonexistent-fixture-remote")
        checkpoint=remote/"diagnostics/diagnostic-onehop-s20260913/checkpoint-416"
        command=[str(remote/".venv/bin/python"),"-m","ouro_depth.train","evaluate","--model-path",str(remote/"base_model"),
                 "--checkpoint",str(checkpoint),"--data-dir",str(remote/"data/v3-pointer"),"--eval-file","dev.jsonl",
                 "--output",str(remote/"artifacts/v3-initializer-dev"),"--eval-batch","8","--depths","4,6,8"]
        value={"command":command,"checkpoint":str(checkpoint),"source":str(remote/"diagnostics/v2-extrapolation-dev/source"),
               "scope":"v3 development initializer only","expected_count":1280,"scored_test":False,"unchanged_evaluator_source_reused":True}
        path=self.root/"artifacts/v3-initializer-launch.json";path.parent.mkdir();path.write_text(json.dumps(value))
        self.assertTrue(validate_initializer_launch(self.root,checkpoint)["registered_command_matches"])
        for field,replacement in [("checkpoint",str(checkpoint.parent/"checkpoint-1")),("source",str(remote/"other")),("scored_test",True)]:
            changed=copy.deepcopy(value);changed[field]=replacement;path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises(ValueError):validate_initializer_launch(self.root,checkpoint)
        value["command"][value["command"].index("--eval-file")+1]="test.jsonl";path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):validate_initializer_launch(self.root,checkpoint)


if __name__ == "__main__":
    unittest.main()
