"""Choose a checkpoint using dev metrics only; never use the fixed test subset."""

import argparse
import json
import math
from pathlib import Path

from .common import read_jsonl, sha256, write_json
from .convert_clevrer import assign_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-dataset', required=True)
    parser.add_argument('--candidate', action='append', nargs=3, metavar=('STEP', 'ADAPTER', 'METRICS'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.dev_dataset)
    if len(rows) != 2000:
        raise ValueError('Historical selection requires 2000 development questions')
    for row in rows:
        index = int(row['video_id'].removeprefix('video_'))
        if not 0 <= index < 10000 or assign_split(row['video_id'], source_role='official_train', seed=42, dev_ratio=0.1) != 'dev':
            raise ValueError('Candidate dataset is not the original video-isolated official-train dev split')
    candidates = []
    for step, adapter, metrics_path in args.candidate:
        metrics = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
        if metrics.get('dataset_sha256') != sha256(args.dev_dataset) or metrics['overall']['count'] != 2000:
            raise ValueError('Candidate metrics must be scored on the declared dev dataset')
        if int(step) <= 0 or not math.isfinite(metrics['overall']['accuracy']) or not 0 <= metrics['overall']['accuracy'] <= 1:
            raise ValueError('Candidate step and accuracy are invalid')
        candidates.append({'step': int(step), 'adapter': adapter, 'accuracy': metrics['overall']['accuracy'],
                           'metrics_sha256': sha256(metrics_path)})
    selected = min(candidates, key=lambda item: (-item['accuracy'], item['step'], item['adapter']))
    write_json(args.output, {'selection_split': 'dev', 'rule': 'max_accuracy_then_lowest_step',
                             'candidates': candidates, 'selected': selected,
                             'warning': 'Historical winner was step 700; a new training run can select a different step.'})
    print(json.dumps(selected))


if __name__ == '__main__':
    main()
