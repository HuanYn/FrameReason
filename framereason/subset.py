"""Recreate the fixed balanced 800-question subset using the frozen identity hash."""

import argparse
import hashlib
import json

from .common import read_jsonl, record_id, sha256, unique_rows, write_json, write_jsonl

TYPES = ('descriptive', 'explanatory', 'predictive', 'counterfactual')


def rank_key(row, seed=42):
    payload = ['clevrer.compact800.sampling/1', seed, row['question_type'], row['video_id'], row['question_id']]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), record_id(row)


def select_subset(rows, per_type=200, seed=42, require_parent=True):
    unique_rows(rows)
    if any(row.get('question_type') not in TYPES for row in rows):
        raise ValueError('unexpected question type')
    result = []
    for question_type in TYPES:
        group = [row for row in rows if row['question_type'] == question_type]
        if len(group) < per_type or (require_parent and len(group) != 1000):
            raise ValueError(f'{question_type}: need exactly 1000 parent rows for the fixed protocol')
        result.extend(sorted(group, key=lambda row: rank_key(row, seed))[:per_type])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-keys', help='Optional published subset800 JSONL; verify exact ordered membership')
    args = parser.parse_args()
    selected = select_subset(read_jsonl(args.dataset))
    if args.expected_keys:
        expected = read_jsonl(args.expected_keys)
        if [record_id(row) for row in selected] != [record_id(row) for row in expected]:
            raise ValueError('Recreated subset membership/order differs from the published fixed800 keys')
    write_jsonl(args.output, selected)
    write_json(args.output + '.manifest.json', {
        'schema': 'framereason.fixed800/1', 'seed': 42,
        'source_sha256': sha256(args.dataset), 'subset_sha256': sha256(args.output),
        'rows': len(selected), 'unique_videos': len({row['video_id'] for row in selected}),
        'record_ids': [record_id(row) for row in selected],
        'note': 'Absolute image paths change byte hashes across machines; compare identity membership separately.',
    })
    print(f'Selected {len(selected)} questions; outputs were written without overwriting existing files.')


if __name__ == '__main__':
    main()
