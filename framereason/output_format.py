"""Strict, deterministic parsing for the public CLEVRER trace format.

The trace is a short symbolic program, not a request for private chain-of-thought.
Only the following envelope is considered format-valid::

    <trace>{...a JSON object...}</trace><answer>...</answer>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

_ENVELOPE_RE = re.compile(r'\A\s*<trace>(?P<trace>.*?)</trace>\s*<answer>(?P<answer>.*?)</answer>\s*\Z', re.DOTALL)
_TRACE_RE = re.compile(r'<trace>(.*?)</trace>', re.DOTALL)
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


@dataclass(frozen=True)
class ParsedOutput:
    """Result of parsing one completion without raising on model output."""

    format_valid: bool
    trace: Optional[Dict[str, Any]]
    answer: Optional[str]
    error: Optional[str]


def _single_tag_payload(pattern: re.Pattern[str], text: str) -> Optional[str]:
    matches = pattern.findall(text)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_non_finite(value: str):
    raise ValueError(f'non-finite JSON value: {value}')


def parse_output(text: str) -> ParsedOutput:
    """Parse one completion.

    A single answer remains available for answer-only scoring when the trace JSON
    is malformed. Repeated/ambiguous tag pairs are never guessed.
    """

    if not isinstance(text, str):
        return ParsedOutput(False, None, None, 'completion_not_string')

    if any(text.count(tag) != 1 for tag in ('<trace>', '</trace>', '<answer>', '</answer>')):
        return ParsedOutput(False, None, None, 'ambiguous_tags')
    answer = _single_tag_payload(_ANSWER_RE, text)
    raw_trace = _single_tag_payload(_TRACE_RE, text)
    if raw_trace is None:
        return ParsedOutput(False, None, answer, 'ambiguous_trace')
    if any(tag in raw_trace for tag in ('<trace>', '</trace>', '<answer>', '</answer>')):
        return ParsedOutput(False, None, answer, 'tag_in_trace')
    if answer is not None and any(tag in answer for tag in ('<trace>', '</trace>', '<answer>', '</answer>')):
        return ParsedOutput(False, None, None, 'tag_in_answer')

    try:
        trace = json.loads(
            raw_trace,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ParsedOutput(False, None, answer, 'invalid_trace_json')
    if not isinstance(trace, dict):
        return ParsedOutput(False, None, answer, 'trace_not_object')
    if answer == '':
        answer = None
    if _ENVELOPE_RE.fullmatch(text) is None:
        return ParsedOutput(False, trace, answer, 'invalid_envelope')
    if answer is None:
        return ParsedOutput(False, trace, None, 'empty_answer')
    return ParsedOutput(True, trace, answer, None)


def render_output(trace: Dict[str, Any], answer: str) -> str:
    """Render the canonical compact target used by SFT and evaluation."""

    if not isinstance(trace, dict):
        raise TypeError('trace must be a dictionary')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('answer must be a non-empty string')
    reserved_tags = ('<trace>', '</trace>', '<answer>', '</answer>')
    if any(tag in answer for tag in reserved_tags):
        raise ValueError('answer must not contain reserved output tags')
    trace_json = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if any(tag in trace_json for tag in reserved_tags):
        raise ValueError('trace must not contain reserved output tags')
    return f'<trace>{trace_json}</trace><answer>{answer.strip()}</answer>'


def has_repetition(text: str, *, ngram_size: int = 4) -> bool:
    """Detect unambiguous output/tag repetition with deterministic rules."""

    if not isinstance(text, str):
        return False
    for tag in ('<trace>', '</trace>', '<answer>', '</answer>'):
        if text.count(tag) > 1:
            return True

    tokens = re.findall(r'[A-Za-z0-9_]+|[\u4e00-\u9fff]+', text.casefold())
    run = 1
    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1

    if ngram_size > 0:
        for start in range(0, len(tokens) - 2 * ngram_size + 1):
            first = tokens[start:start + ngram_size]
            second = tokens[start + ngram_size:start + 2 * ngram_size]
            if first == second:
                return True
    return False
