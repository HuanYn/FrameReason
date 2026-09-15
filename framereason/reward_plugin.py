"""ms-swift external ORM adapter for the dependency-free reward core."""

from __future__ import annotations

import os
import sys
from typing import Any, List

from swift.rewards import ORM, orms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from reward_core import score_completion, trace_is_consistent  # noqa: E402
from output_format import parse_output  # noqa: E402


def _required_batch(kwargs: dict, name: str, size: int) -> List[Any]:
    value = kwargs.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'CLEVRER reward requires batched column {name!r} with length {size}')
    return list(value)


class CLEVRERVerifiableReward(ORM):
    """Return the single fixed weighted score for every generated completion."""

    def __call__(self, completions, **kwargs) -> List[float]:
        size = len(completions)
        solutions = _required_batch(kwargs, 'solution', size)
        question_types = _required_batch(kwargs, 'question_type', size)
        programs = _required_batch(kwargs, 'program', size)
        reward_metadata = _required_batch(kwargs, 'reward_metadata', size)
        max_chars = kwargs.get('max_completion_chars', [4096] * size)
        if not isinstance(max_chars, (list, tuple)):
            max_chars = [max_chars] * size
        if len(max_chars) != size:
            raise ValueError('max_completion_chars must be scalar or match completions length')

        rewards = []
        for index, completion in enumerate(completions):
            reference = parse_output(solutions[index])
            if not reference.format_valid or reference.answer is None:
                raise ValueError(f'CLEVRER solution at batch index {index} violates the canonical output contract')
            if not trace_is_consistent(
                    reference.trace,
                    question_type=question_types[index],
                    program=programs[index],
                    reward_metadata=reward_metadata[index]):
                raise ValueError(
                    f'CLEVRER solution at batch index {index} disagrees with program/reward_metadata columns')
            breakdown = score_completion(
                completion,
                reference_answer=reference.answer,
                question_type=question_types[index],
                program=programs[index],
                reward_metadata=reward_metadata[index],
                max_completion_chars=max_chars[index],
            )
            rewards.append(breakdown.total)
        return rewards


orms['clevrer_verifiable'] = CLEVRERVerifiableReward
