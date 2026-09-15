"""Build a sanitized release from existing archives; no network or model execution.

This is a maintainer utility. End users normally run verify_results.py instead.
Only explicitly allowlisted fields are exported. Original inputs are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROLES = ("base", "sft", "grpo")
TYPES = ("descriptive", "explanatory", "predictive", "counterfactual")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if match is None:
        raise ValueError("Archived response has no answer tag")
    return match.group(1).strip()


def write(path, data, jsonl=False):
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in data) if jsonl else json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base_predictions", "sft_predictions", "grpo_predictions", "base_metrics", "sft_metrics", "grpo_extract", "base_failures", "sft_failures", "extra_records", "selected_records", "completions", "scalars"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions = {role: lines(getattr(args, role + "_predictions")) for role in ROLES}
    extract = load(args.grpo_extract)
    metrics = {"base": load(args.base_metrics), "sft": load(args.sft_metrics), "grpo": extract["artifacts/metrics.json"]["data"]}
    extra = load(args.extra_records)
    assert extra["grpoFailureLedger"]["sha256"] == extract["failuresSource"]["sha256"]
    failures = {"base": lines(args.base_failures), "sft": lines(args.sft_failures), "grpo": extra["grpoFailureLedger"]["rows"]}
    for role in ROLES:
        assert len(predictions[role]) == 800
        assert sha(getattr(args, role + "_predictions")) == metrics[role]["predictions_sha256"]
        assert len(failures[role]) == metrics[role]["failure_count"]
    for role in ("base", "sft"):
        assert sha(getattr(args, role + "_failures")) == metrics[role]["failures_sha256"]
    failure_maps = {role: {row["ordinal"]: row for row in failures[role]} for role in ROLES}
    identity = metrics["base"]["binding"]["dataset_audit"]["identity_records"]
    paired = []
    subset = []
    for ordinal, row in enumerate(predictions["base"]):
        keys = {k: row[k] for k in ("video_id", "question_id", "question_type", "record_id")}
        assert keys == identity[ordinal]
        assert hashlib.sha256(f'{row["video_id"]}:{row["question_id"]}'.encode()).hexdigest() == row["record_id"]
        item = dict(ordinal=ordinal, **keys, models={})
        subset.append(dict(ordinal=ordinal, **keys))
        for role in ROLES:
            pred = predictions[role][ordinal]
            assert pred["ordinal"] == ordinal
            assert {k: pred[k] for k in keys} == keys
            failure = failure_maps[role].get(ordinal)
            if failure:
                assert all(failure[k] == row[k] for k in ("record_id", "video_id", "question_id"))
            item["models"][role] = {"response": pred["response"], "answer": answer(pred["response"]), "answer_correct": failure["answer_correct"] if failure else True}
        paired.append(item)
    assert len({r["record_id"] for r in subset}) == 800
    assert len({r["video_id"] for r in subset}) == 754
    write(out / "predictions.jsonl", paired, True)
    write(out / "subset800.jsonl", subset, True)
    known = load(args.selected_records)["records"] + extra["records"]
    references = []
    for item in known:
        row = item["record"]
        refs = {k: row[k] for k in ("video_id", "question_id", "question_type", "solution")}
        refs["answer"] = answer(row["solution"])
        refs["messages"] = row["messages"]
        references.append(refs)
    assert len(references) == 10
    write(out / "demo-references.jsonl", references, True)
    audit = metrics["base"]["binding"]["dataset_audit"]
    write(out / "sampling-spec.json", audit["sampling_spec"])
    scalar_rows = lines(args.scalars)
    scalar_steps = {int(x["global_step/max_steps"].split("/")[0]): x for x in scalar_rows if "grad_norm" in x}
    completion_rows = lines(args.completions)
    assert len(completion_rows) == 3001 and completion_rows[-1] == completion_rows[-2]
    groups = []
    for row in completion_rows[:-1]:
        steps = {int(x) for x in row["step"]}
        assert len(steps) == 1
        step = steps.pop()
        assert len(row["CLEVRERVerifiableReward"]) == len(row["advantages"]) == 4
        groups.append({"step": step, "rewards": row["CLEVRERVerifiableReward"], "advantages": row["advantages"], "grad_norm": scalar_steps[step]["grad_norm"]})
    assert {x["step"] for x in groups} == set(range(1, 3001))
    write(out / "training-groups.jsonl", groups, True)
    same = [row for row in groups if len(set(round(x, 6) for x in row["rewards"])) == 1]
    assert len(same) == 2659
    assert all(not any(row["advantages"]) for row in same)
    summary = {
        "schema_version": "framereason.public_results/1",
        "status": "historical_single_seed_results_not_new_inference",
        "dataset": {"name": "CLEVRER custom fixed800", "official_source_split": "validation", "parent_rows": 4000, "evaluated_rows": 800, "unique_videos": 754, "rows_per_type": 200, "train_rows": 12000, "development_rows": 2000, "frames_per_question": 8, "seed": 42, "subset_sha256_historical_path_bound_jsonl": audit["dataset_sha256"], "parent_sha256_historical_path_bound_jsonl": audit["parent_dataset_sha256"]},
        "models": {}, "paired": {},
        "training_signal": {"unique_groups": 3000, "candidates_per_group": 4, "equal_reward_groups": len(same), "equal_reward_fraction": len(same) / 3000, "unequal_reward_groups": 3000 - len(same), "equal_reward_nonzero_advantage_groups": 0, "zero_grad_norm_steps": sum(row["grad_norm"] == 0 for row in groups), "equal_reward_histogram": dict(sorted(Counter(str(round(row["rewards"][0], 6)) for row in same).items())), "deduplication": "Original completion log has 3001 lines; only the final exact duplicate is excluded in this derived view.", "comparison_precision_decimal_places": 6},
        "decode": metrics["base"]["decode_protocol"],
        "reproduction_scope": {"all_800_raw_responses_included": True, "all_800_historical_answer_correct_flags_included": True, "all_800_official_ground_truth_included": False, "official_reference_examples_included": 10, "weights_included": False, "full_training_dataset_included": False, "default_verification": "Recount archived per-question correctness and logged rewards; not an independent full official-ground-truth re-score."},
    }
    for role in ROLES:
        m = metrics[role]
        count = sum(r["models"][role]["answer_correct"] for r in paired)
        assert count == m["overall"]["answer_correct"]["numerator"]
        by_type = {kind: {"correct": sum(r["models"][role]["answer_correct"] for r in paired if r["question_type"] == kind), "total": 200} for kind in TYPES}
        for kind in TYPES:
            assert by_type[kind]["correct"] == m["by_question_type"][kind]["answer_correct"]["numerator"]
        summary["models"][role] = {"checkpoint_step": {"base": None, "sft": 700, "grpo": 3000}[role], "answer_correct": count, "total": 800, "answer_accuracy": count / 800, "format_valid": m["overall"]["format_valid"]["numerator"], "full_trace_consistent": m["overall"]["full_trace_consistent"]["numerator"], "joint_success": m["overall"]["joint_success"]["numerator"], "mean_rule_reward": m["overall"]["reward"]["mean"], "by_type": by_type}
    for target, baseline in (("sft", "base"), ("grpo", "sft"), ("grpo", "base")):
        counts = Counter((r["models"][baseline]["answer_correct"], r["models"][target]["answer_correct"]) for r in paired)
        net = counts[False, True] - counts[True, False]
        summary["paired"][f"{target}_vs_{baseline}"] = {"corrected": counts[False, True], "lost": counts[True, False], "both_correct": counts[True, True], "both_wrong": counts[False, False], "net_correct": net, "delta_percentage_points": net / 800 * 100, "relative_accuracy_change": net / summary["models"][baseline]["answer_correct"]}
    assert summary["paired"]["grpo_vs_sft"]["corrected"] == 59
    assert summary["paired"]["grpo_vs_sft"]["lost"] == 62
    write(out / "metrics.json", summary)
    provenance = {
        "schema_version": "framereason.public_provenance/1",
        "original_artifact_sha256": {key: sha(value) for key, value in vars(args).items() if key != "output_dir"},
        "grpo_original_metrics_sha256": extract["artifacts/metrics.json"]["source"]["sha256"],
        "grpo_original_failures_sha256": extract["failuresSource"]["sha256"],
        "published_artifact_sha256": {p.name: sha(p) for p in (out / name for name in ("predictions.jsonl", "subset800.jsonl", "demo-references.jsonl", "training-groups.jsonl", "sampling-spec.json", "metrics.json"))},
        "sanitization": "Allowlisted scientific fields only. No hostnames, credentials, user filesystem paths, GPU UUIDs, weights, optimizer states, or frame bytes.",
        "historical_versions": {"model_repository": "Qwen/Qwen3-VL-4B-Instruct", "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17", "training_code_commit": "89e4bcda66583b43fdb5e35cc5dca9cee241f123", "evaluation_protocol_commit": "fee1c74e33335838ca836384fb8c527bc856a8fe", "recovery_execution_commit": "ff3591b0fd916b85af50fb0580be0af4b1e685a1", "python": "3.11.11", "torch": "2.6.0+cu124", "transformers": "4.57.3", "ms_swift": "4.5.2", "peft": "0.19.1", "trl": "0.29.1"},
        "adapter_manifest_sha256": {"sft700": "d6894ba1d67c3fd4fba1732c1228376b125ff865520ab285c4512f9b9004c0f2", "grpo3000": "d68c75311b6f1adcaf19c714c644ff2b20d48019565059f22060e9887656390f"},
    }
    write(out / "provenance.json", provenance)
    print(json.dumps({"status": "EXPORTED", "prediction_rows": len(paired), "training_groups": len(groups), "answer_correct": {r: summary["models"][r]["answer_correct"] for r in ROLES}}))


if __name__ == "__main__":
    main()
