"""Select the exact historical 800 question IDs from a converted CLEVRER test set.

Unlike reranking a newly built parent set, this reproduces the published question
membership and stored prediction order exactly. Images retain the caller's paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def select_rows(rows, identities):
    indexed = {}
    for row in rows:
        key = (row["video_id"], row["question_id"])
        if type(key[1]) is not int or key[1] < 0 or key in indexed:
            raise ValueError(f"Invalid or duplicate question identity: {key}")
        indexed[key] = row
    selected = []
    for ordinal, identity in enumerate(identities):
        if identity["ordinal"] != ordinal:
            raise ValueError("Subset identity file is not in contiguous ordinal order")
        key = (identity["video_id"], identity["question_id"])
        digest = hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest()
        if identity["record_id"] != digest:
            raise ValueError(f"Invalid subset record ID: {key}")
        if key not in indexed:
            raise ValueError(f"Historical question missing from provided test set: {key}")
        row = indexed[key]
        if row["question_type"] != identity["question_type"]:
            raise ValueError(f"Question type changed: {key}")
        selected.append(row)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Converted, labeled test JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identities", type=Path, default=Path(__file__).resolve().with_name("subset800.jsonl"))
    args = parser.parse_args()
    identities = read_jsonl(args.identities)
    if len(identities) != 800 or len({row["record_id"] for row in identities}) != 800:
        raise ValueError("Expected exactly 800 unique historical question identities")
    selected = select_rows(read_jsonl(args.input), identities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"selected_rows": len(selected), "unique_videos": len({row["video_id"] for row in selected}), "order": "published historical prediction ordinal", "note": "Membership reproduced; serialized bytes may differ because caller media paths differ."}))


if __name__ == "__main__":
    main()
