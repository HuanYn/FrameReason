"""Dependency-free deterministic reward used by the CLEVRER GRPO plugin."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

try:
    from .output_format import has_repetition, parse_output
except ImportError:  # Enables loading reward_plugin.py by absolute path.
    from output_format import has_repetition, parse_output

TRACE_KEYS = frozenset({
    'question_type',
    'question_program',
    'selected_choice_ids',
    'selected_choice_programs',
})
REWARD_METADATA_KEYS = frozenset({'answer_kind', 'correct_choice_ids', 'selected_choice_programs'})
ANSWER_KINDS = frozenset({'short_text', 'choice_ids'})
QUESTION_TYPES = frozenset({'descriptive', 'explanatory', 'predictive', 'counterfactual'})


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable terms for the fixed scalar reward."""

    total: float
    format_reward: float
    answer_reward: float
    consistency_reward: float
    penalty: float
    answer_correct: bool
    trace_consistent: bool
    repeated: bool
    overlength: bool
    parse_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(
                value,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return fallback
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = item
    return result


def _reject_non_finite(value):
    raise ValueError(f'non-finite JSON value: {value}')


def _valid_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_ids(value: Any) -> Optional[List[int]]:
    value = _json_value(value, None)
    if not isinstance(value, (list, tuple)) or not all(_valid_int(item) and item >= 0 for item in value):
        return None
    result = list(value)
    if result != sorted(set(result)):
        return None
    return result


def canonical_program(value: Any) -> Optional[List[str]]:
    """Canonicalize the official non-empty CLEVRER postfix token list."""

    value = _json_value(value, None)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        return None
    return list(value)


def _canonical_choice_programs(value: Any) -> Optional[List[Dict[str, Any]]]:
    value = _json_value(value, None)
    if not isinstance(value, (list, tuple)):
        return None
    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {'choice_id', 'program'}:
            return None
        choice_id = item['choice_id']
        program = canonical_program(item['program'])
        if not _valid_int(choice_id) or choice_id < 0 or program is None:
            return None
        result.append({'choice_id': choice_id, 'program': program})
    if [item['choice_id'] for item in result] != sorted({item['choice_id'] for item in result}):
        return None
    return result


def canonical_reward_metadata(value: Any) -> Optional[Dict[str, Any]]:
    """Validate the private reward metadata emitted by the converter."""

    value = _json_value(value, None)
    if not isinstance(value, Mapping) or set(value) != REWARD_METADATA_KEYS:
        return None
    answer_kind = value.get('answer_kind')
    ids = _canonical_ids(value.get('correct_choice_ids'))
    choice_programs = _canonical_choice_programs(value.get('selected_choice_programs'))
    if answer_kind not in ANSWER_KINDS or ids is None or choice_programs is None:
        return None
    if answer_kind == 'short_text' and (ids or choice_programs):
        return None
    if answer_kind == 'choice_ids' and [item['choice_id'] for item in choice_programs] != ids:
        return None
    return {
        'answer_kind': answer_kind,
        'correct_choice_ids': ids,
        'selected_choice_programs': choice_programs,
    }


def build_reference_trace(*, question_type: str, program: Any, reward_metadata: Any) -> Optional[Dict[str, Any]]:
    """Build the exact public trace contract from private annotation metadata."""

    canonical_question_program = canonical_program(program)
    canonical_metadata = canonical_reward_metadata(reward_metadata)
    if question_type not in QUESTION_TYPES or canonical_question_program is None or canonical_metadata is None:
        return None
    expected_answer_kind = 'short_text' if question_type == 'descriptive' else 'choice_ids'
    if canonical_metadata['answer_kind'] != expected_answer_kind:
        return None
    return {
        'question_type': question_type,
        'question_program': canonical_question_program,
        'selected_choice_ids': canonical_metadata['correct_choice_ids'],
        'selected_choice_programs': canonical_metadata['selected_choice_programs'],
    }


def trace_is_consistent(trace: Any, *, question_type: str, program: Any, reward_metadata: Any) -> bool:
    """Require exact schema, types, program and selected-choice evidence."""

    if not isinstance(trace, Mapping) or set(trace) != TRACE_KEYS:
        return False
    expected = build_reference_trace(question_type=question_type, program=program, reward_metadata=reward_metadata)
    expected_metadata = canonical_reward_metadata(reward_metadata)
    if expected is None or expected_metadata is None:
        return False
    trace_metadata = {
        'answer_kind': expected_metadata['answer_kind'],
        'correct_choice_ids': trace.get('selected_choice_ids'),
        'selected_choice_programs': trace.get('selected_choice_programs'),
    }
    generated = build_reference_trace(
        question_type=trace.get('question_type'),
        program=trace.get('question_program'),
        reward_metadata=trace_metadata,
    )
    return generated is not None and generated == expected


def _choice_answer_ids(answer: str) -> Optional[List[int]]:
    stripped = answer.strip()
    if not stripped.startswith('[') or not stripped.endswith(']'):
        return None
    parsed = _json_value(stripped, None)
    return _canonical_ids(parsed)


def answer_is_correct(predicted: Optional[str], reference_answer: str, *, answer_kind: str) -> bool:
    if predicted is None or not isinstance(reference_answer, str):
        return False
    if answer_kind == 'choice_ids':
        expected_ids = _choice_answer_ids(reference_answer)
        return expected_ids is not None and _choice_answer_ids(predicted) == expected_ids
    if answer_kind != 'short_text':
        return False

    def normalize(value):
        return ' '.join(value.strip().casefold().split())

    return normalize(predicted) == normalize(reference_answer)


def score_completion(
    completion: str,
    *,
    reference_answer: str,
    question_type: str,
    program: Any,
    reward_metadata: Any,
    max_completion_chars: int = 4096,
) -> RewardBreakdown:
    """Compute ``0.1 format + 0.5 answer + 0.4 program exact-match - 0.1 penalty``.

    Repetition and overlength share one capped 0.1 penalty. The final value is
    clamped to [0, 1]. All operations are local and deterministic.
    """

    if not isinstance(max_completion_chars, int) or isinstance(max_completion_chars, bool) or max_completion_chars <= 0:
        raise ValueError('max_completion_chars must be a positive integer')
    canonical_metadata = canonical_reward_metadata(reward_metadata)
    parsed = parse_output(completion)
    answer_correct = canonical_metadata is not None and answer_is_correct(
        parsed.answer,
        reference_answer,
        answer_kind=canonical_metadata['answer_kind'],
    )
    trace_consistent = parsed.trace is not None and trace_is_consistent(
        parsed.trace,
        question_type=question_type,
        program=program,
        reward_metadata=reward_metadata,
    )
    repeated = has_repetition(completion)
    overlength = isinstance(completion, str) and len(completion) > max_completion_chars

    format_reward = 0.1 if parsed.format_valid else 0.0
    answer_reward = 0.5 if answer_correct else 0.0
    consistency_reward = 0.4 if trace_consistent else 0.0
    penalty = 0.1 if repeated or overlength else 0.0
    raw_total = format_reward + answer_reward + consistency_reward - penalty
    total = round(min(1.0, max(0.0, raw_total)), 10)
    if not math.isfinite(total):
        total = 0.0
    return RewardBreakdown(
        total=total,
        format_reward=format_reward,
        answer_reward=answer_reward,
        consistency_reward=consistency_reward,
        penalty=penalty,
        answer_correct=answer_correct,
        trace_consistent=trace_consistent,
        repeated=repeated,
        overlength=overlength,
        parse_error=parsed.error,
    )
