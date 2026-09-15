"""Small CPU regression tests for the published evidence helpers."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_subset import select_rows
from verify_results import check_references, normalize_answer, read_jsonl, verify

HERE = Path(__file__).resolve().parent


class PublishedResultsTests(unittest.TestCase):
    def test_complete_published_recount(self):
        result = verify(HERE)
        self.assertEqual(result["answer_correct"], {"base": 343, "sft": 609, "grpo": 606})
        self.assertFalse(result["full_ground_truth_rescore_completed"])

    def test_answer_normalization_rejects_invalid_choice_lists(self):
        for text in ("[1,0]", "[0,0]", "[true]", "[-1]", '"0"', "not-json"):
            self.assertIsNone(normalize_answer(text, "counterfactual"))
        self.assertEqual(normalize_answer("[0, 2]", "counterfactual"), [0, 2])
        self.assertEqual(normalize_answer("  Metal  Cylinder ", "descriptive"), "metal cylinder")

    def test_real_ten_reference_rescore_and_missing_full_reference(self):
        predictions = read_jsonl(HERE / "predictions.jsonl")
        references = HERE / "demo-references.jsonl"
        self.assertEqual(check_references(predictions, references, False), 10)
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            check_references(predictions, references, True)
        keys = {(row["video_id"], row["question_id"]) for row in read_jsonl(references)}
        matching = [row for row in predictions if (row["video_id"], row["question_id"]) in keys]
        self.assertEqual(check_references(matching, references, True), 10)

    def test_tampered_saved_correctness_is_detected(self):
        rows = copy.deepcopy(read_jsonl(HERE / "predictions.jsonl"))
        rows[0]["models"]["base"]["answer_correct"] = not rows[0]["models"]["base"]["answer_correct"]
        with self.assertRaisesRegex(ValueError, "Reference rescore differs"):
            check_references(rows, HERE / "demo-references.jsonl", False)

    def test_duplicate_reference_rejected(self):
        row = read_jsonl(HERE / "demo-references.jsonl")[0]
        with tempfile.TemporaryDirectory(prefix="framereason-reference-test-") as temp:
            path = Path(temp) / "duplicate.jsonl"
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate reference"):
                check_references(read_jsonl(HERE / "predictions.jsonl"), path, False)

    def test_subset_selects_exact_identity_not_input_order(self):
        identities = read_jsonl(HERE / "subset800.jsonl")
        rows = [{key: row[key] for key in ("video_id", "question_id", "question_type")} for row in identities]
        self.assertEqual(select_rows(list(reversed(rows)), identities), rows)
        with self.assertRaisesRegex(ValueError, "missing"):
            select_rows(rows[:-1], identities)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_rows(rows + rows[:1], identities)
        altered = copy.deepcopy(identities)
        altered[0]["record_id"] = hashlib.sha256(b"not-the-record").hexdigest()
        with self.assertRaisesRegex(ValueError, "record ID"):
            select_rows(rows, altered)


if __name__ == "__main__":
    unittest.main()
