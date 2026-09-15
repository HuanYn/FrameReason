"""CPU-only checks of published historical results (Python standard library).

Default: recount archived correctness flags and rewards, and rescore the 10
included official references. This is NOT a fresh 800-question model evaluation
or an independent full official-ground-truth rescore.

Optional --reference-jsonl accepts the converted, labeled CLEVRER test JSONL
and independently checks all 2400 saved answers against its solutions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROLES = ("base", "sft", "grpo")
TYPES = ("descriptive", "explanatory", "predictive", "counterfactual")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_text(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    check(len(matches) == 1, "Expected exactly one answer field")
    return matches[0].strip()


def normalize_answer(answer, question_type):
    if question_type == "descriptive":
        return " ".join(answer.strip().lower().split())
    try:
        ids = json.loads(answer)
    except (ValueError, TypeError):
        return None
    if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
        return None
    if ids != sorted(set(ids)):
        return None
    return ids


def check_references(predictions, path, require_all):
    references = {}
    for row in read_jsonl(path):
        key = (row["video_id"], row["question_id"])
        check(key not in references, f"Duplicate reference key: {key}")
        references[key] = row
    matched = 0
    for row in predictions:
        key = (row["video_id"], row["question_id"])
        reference = references.get(key)
        if reference is None:
            check(not require_all, f"Missing reference: {key}")
            continue
        check(reference["question_type"] == row["question_type"], f"Reference type mismatch: {key}")
        truth_text = answer_text(reference["solution"])
        truth = normalize_answer(truth_text, row["question_type"])
        check(truth is not None, f"Invalid reference answer: {key}")
        for role in ROLES:
            model = row["models"][role]
            predicted = normalize_answer(answer_text(model["response"]), row["question_type"])
            correct = predicted is not None and predicted == truth
            check(correct == model["answer_correct"], f"Reference rescore differs: {role}, {key}")
        matched += 1
    check(matched > 0, "No published prediction matches the supplied references")
    return matched


def verify(directory, reference_jsonl=None):
    metrics = load(directory / "metrics.json")
    provenance = load(directory / "provenance.json")
    for name, digest in provenance["published_artifact_sha256"].items():
        check(Path(name).name == name, "Unsafe published artifact name")
        check(sha256(directory / name) == digest, f"SHA256 mismatch: {name}")
    predictions = read_jsonl(directory / "predictions.jsonl")
    subset = read_jsonl(directory / "subset800.jsonl")
    groups = read_jsonl(directory / "training-groups.jsonl")
    check(len(predictions) == len(subset) == 800, "Expected exactly 800 questions")
    check(len({row["record_id"] for row in predictions}) == 800, "Duplicate record IDs")
    check(len({row["video_id"] for row in predictions}) == 754, "Expected 754 unique videos")
    check(Counter(row["question_type"] for row in predictions) == {kind: 200 for kind in TYPES}, "Unbalanced question types")
    for ordinal, row in enumerate(predictions):
        check(row["ordinal"] == ordinal, "Prediction order mismatch")
        check(re.fullmatch(r"video_\d{5}", row["video_id"]) is not None, "Invalid video ID")
        check(type(row["question_id"]) is int and row["question_id"] >= 0, "Invalid question ID")
        expected_id = hashlib.sha256(f'{row["video_id"]}:{row["question_id"]}'.encode()).hexdigest()
        check(row["record_id"] == expected_id, "Invalid record ID")
        check(subset[ordinal] == {key: value for key, value in row.items() if key != "models"}, "Subset identity differs from predictions")
        check(set(row["models"]) == set(ROLES), "Expected three model outputs")
        for role in ROLES:
            model = row["models"][role]
            check(type(model["answer_correct"]) is bool, "Correctness must be an archived boolean, not a weight or score")
            check(model["answer"] == answer_text(model["response"]), "Displayed answer differs from raw model response")
    accuracy = {}
    for role in ROLES:
        count = sum(row["models"][role]["answer_correct"] for row in predictions)
        accuracy[role] = count
        claimed = metrics["models"][role]
        check(claimed["answer_correct"] == count and claimed["total"] == 800, "Aggregate answer count mismatch")
        check(claimed["answer_accuracy"] == count / 800, "Answer accuracy mismatch")
        for kind in TYPES:
            type_correct = sum(row["models"][role]["answer_correct"] for row in predictions if row["question_type"] == kind)
            check(claimed["by_type"][kind] == {"correct": type_correct, "total": 200}, "Per-type answer count mismatch")
    for target, baseline in (("sft", "base"), ("grpo", "sft"), ("grpo", "base")):
        paired = Counter((row["models"][baseline]["answer_correct"], row["models"][target]["answer_correct"]) for row in predictions)
        actual = {"corrected": paired[False, True], "lost": paired[True, False], "both_correct": paired[True, True], "both_wrong": paired[False, False]}
        claimed = metrics["paired"][f"{target}_vs_{baseline}"]
        check(all(claimed[key] == value for key, value in actual.items()), "Paired count mismatch")
        net = actual["corrected"] - actual["lost"]
        check(claimed["net_correct"] == net, "Paired net mismatch")
        check(math.isclose(claimed["delta_percentage_points"], net / 800 * 100), "Percentage-point mismatch")
        check(math.isclose(claimed["relative_accuracy_change"], net / accuracy[baseline]), "Relative accuracy mismatch")
    check(len(groups) == 3000 and {row["step"] for row in groups} == set(range(1, 3001)), "Missing or duplicate training steps")
    for row in groups:
        check(len(row["rewards"]) == len(row["advantages"]) == 4, "Expected four candidates per training group")
        check(all(math.isfinite(value) for value in row["rewards"] + row["advantages"] + [row["grad_norm"]]), "Non-finite training value")
    equal = [row for row in groups if len({round(value, 6) for value in row["rewards"]}) == 1]
    signal = metrics["training_signal"]
    histogram = dict(sorted(Counter(str(round(row["rewards"][0], 6)) for row in equal).items()))
    check(len(equal) == signal["equal_reward_groups"] == 2659, "Equal-reward count mismatch")
    check(len(groups) - len(equal) == signal["unequal_reward_groups"], "Unequal-reward count mismatch")
    check(len(equal) / len(groups) == signal["equal_reward_fraction"], "Equal-reward fraction mismatch")
    check(sum(any(row["advantages"]) for row in equal) == signal["equal_reward_nonzero_advantage_groups"] == 0, "Nonzero advantages in an equal-reward group")
    check(sum(row["grad_norm"] == 0 for row in groups) == signal["zero_grad_norm_steps"] == 2659, "Zero-gradient count mismatch")
    check(histogram == signal["equal_reward_histogram"], "Equal-reward histogram mismatch")
    demo_references = check_references(predictions, directory / "demo-references.jsonl", False)
    check(demo_references == 10, "Expected 10 included official reference records")
    optional_references = check_references(predictions, reference_jsonl, True) if reference_jsonl else 0
    return {
        "status": "PASS_WITH_DECLARED_SCOPE",
        "scope": "Historical correctness-ledger recount, exact identity alignment, saved-answer consistency, file hashes, and training-reward statistics; no model execution.",
        "questions": 800, "unique_videos": 754, "answer_correct": accuracy,
        "grpo_vs_sft": metrics["paired"]["grpo_vs_sft"],
        "equal_reward_groups": len(equal), "training_groups": len(groups),
        "included_official_references_rescored": demo_references,
        "user_supplied_official_references_rescored": optional_references,
        "full_ground_truth_rescore_completed": optional_references == 800,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reference-jsonl", type=Path, help="Converted labeled test JSONL containing all 800 keys; enables independent answer-only rescore")
    args = parser.parse_args()
    print(json.dumps(verify(args.results_dir, args.reference_jsonl), indent=2))


if __name__ == "__main__":
    main()
