"""CPU-only exact-answer and annotation-program scoring; not a physics executor."""

import argparse
from collections import defaultdict

from .common import read_jsonl, record_id, sha256, unique_rows, write_json
from .output_format import parse_output
from .reward_core import score_completion, trace_is_consistent


def evaluate(rows, predictions):
    references = unique_rows(rows)
    generated = unique_rows(predictions)
    if set(references) != set(generated):
        raise ValueError('prediction membership differs from reference membership; refusing partial scoring')
    scored = []
    for row in rows:
        identity = record_id(row)
        prediction = generated[identity]
        if prediction.get('record_id', identity) != identity:
            raise ValueError('prediction record_id disagrees with its video/question identity')
        if prediction.get('question_type', row['question_type']) != row['question_type']:
            raise ValueError('prediction question type mismatch')
        output = prediction.get('response')
        if not isinstance(output, str):
            raise ValueError('prediction response must be a string')
        reference = parse_output(row['solution'])
        if not reference.format_valid or not trace_is_consistent(
            reference.trace, question_type=row['question_type'], program=row['program'], reward_metadata=row['reward_metadata']
        ):
            raise ValueError('reference solution is malformed or disagrees with annotation metadata')
        score = score_completion(
            output, reference_answer=reference.answer, question_type=row['question_type'],
            program=row['program'], reward_metadata=row['reward_metadata'],
            max_completion_chars=row.get('max_completion_chars', 4096),
        )
        scored.append({'record_id': identity, 'video_id': row['video_id'], 'question_id': row['question_id'],
                       'question_type': row['question_type'], **score.to_dict()})
    grouped = defaultdict(list)
    for score in scored:
        grouped[score['question_type']].append(score)

    def summarize(group):
        count = len(group)
        correct = sum(item['answer_correct'] for item in group)
        return {'count': count, 'answer_correct': correct, 'accuracy': correct / count,
                'format_valid_rate': sum(item['format_reward'] > 0 for item in group) / count,
                'program_exact_match_rate': sum(item['trace_consistent'] for item in group) / count,
                'mean_reward': sum(item['total'] for item in group) / count}

    return {'overall': summarize(scored), 'by_type': {key: summarize(group) for key, group in grouped.items()},
            'examples': scored, 'event_executor': False, 'llm_judge': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Use grpo_dev/grpo_test rows, not model-visible answers')
    parser.add_argument('--predictions', required=True, help='JSONL with video_id, question_id, response')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = evaluate(read_jsonl(args.dataset), read_jsonl(args.predictions))
    report.update(dataset_sha256=sha256(args.dataset), predictions_sha256=sha256(args.predictions))
    write_json(args.output, report)
    print(report['overall'])


if __name__ == '__main__':
    main()
