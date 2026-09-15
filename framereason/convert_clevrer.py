"""Convert official CLEVRER annotations into auditable ms-swift JSONL.

This command never downloads data. Official training videos are split into
train/dev by a seeded video-id hash; official validation videos are held out as
test. Question caps are applied only after that video-level split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from .output_format import render_output
    from .reward_core import build_reference_trace, canonical_program
except ImportError:  # Supports ``python path/to/convert_clevrer.py``.
    from output_format import render_output
    from reward_core import build_reference_trace, canonical_program

SYSTEM_PROMPT = ('You solve CLEVRER video questions with a public symbolic trace. '
                 'Return exactly <trace>{JSON}</trace><answer>...</answer>. '
                 'The JSON keys must be question_type, question_program, '
                 'selected_choice_ids, and selected_choice_programs. Do not add prose.')
SPLIT_NAMES = ('train', 'dev', 'test')
QUESTION_TYPES = ('descriptive', 'explanatory', 'predictive', 'counterfactual')
SOURCE_ROLES = ('official_train', 'official_validation')
VIDEO_ID_RE = re.compile(r'video_(\d{5})')
DESCRIPTIVE_SUBTYPES = frozenset({'count', 'exist', 'query_color', 'query_material', 'query_shape'})


@dataclass(frozen=True)
class SplitCaps:
    train: int = 3000
    dev: int = 500
    test: int = 1000

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} cap must be a non-negative integer')
        if not any(asdict(self).values()):
            raise ValueError('at least one split cap must be positive')


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_non_finite(value):
    raise ValueError(f'non-finite JSON value: {value}')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_fraction(namespace: str, value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f'{namespace}:{seed}:{value}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def assign_split(video_id: str, *, source_role: str, seed: int, dev_ratio: float) -> str:
    """Assign a whole video while preserving the official validation holdout."""

    if not isinstance(video_id, str) or not video_id:
        raise ValueError('video_id must be a non-empty string')
    if source_role not in SOURCE_ROLES:
        raise ValueError(f'unknown source role: {source_role}')
    if not math.isfinite(dev_ratio) or not 0 < dev_ratio < 1:
        raise ValueError('dev_ratio must be finite and strictly between 0 and 1')
    if source_role == 'official_validation':
        return 'test'
    return 'dev' if _hash_fraction('split', video_id, seed=seed) < dev_ratio else 'train'


def planned_frame_paths(frames_root: Path, video_id: str, frame_count: int) -> List[str]:
    if frame_count <= 0:
        raise ValueError('frame_count must be positive')
    directory = (frames_root / f'{frame_count}f' / video_id).resolve()
    return [str(directory / f'frame_{index:03d}.jpg') for index in range(frame_count)]


def load_annotation_file(path: Path) -> List[Mapping[str, Any]]:
    """Load an official JSON array/object or a scene-per-line JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig') as handle:
        first = ''
        while not first:
            char = handle.read(1)
            if not char:
                return []
            if not char.isspace():
                first = char
        handle.seek(0)
        if first == '[' or (first == '{' and path.suffix.casefold() == '.json'):
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
            if isinstance(payload, Mapping):
                payload = payload.get('scenes')
            if not isinstance(payload, list):
                raise ValueError(f'{path}: top-level JSON must be a list or an object with a scenes list')
            if not all(isinstance(row, Mapping) for row in payload):
                raise ValueError(f'{path}: every scene must be an object')
            return payload
        rows = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
            if not isinstance(row, Mapping):
                raise ValueError(f'{path}:{line_number}: each JSONL row must be an object')
            rows.append(row)
        return rows


def iter_annotation_scenes(train_paths: Sequence[Path],
                           validation_paths: Sequence[Path]) -> Iterator[Tuple[Path, str, Mapping[str, Any]]]:
    for source_role, paths in (
        ('official_train', train_paths),
        ('official_validation', validation_paths),
    ):
        for path in paths:
            for scene in load_annotation_file(path):
                yield path, source_role, scene


def _safe_video_filename(scene: Mapping[str, Any]) -> PurePosixPath:
    filename = scene.get('video_filename')
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('scene is missing video_filename')
    normalized = PurePosixPath(filename.strip().replace('\\', '/'))
    if (normalized.is_absolute() or '..' in normalized.parts or ':' in normalized.parts[0]
            or normalized.name in {'', '.', '..'}):
        raise ValueError(f'unsafe video_filename: {filename!r}')
    return normalized


def _canonical_video_id(scene: Mapping[str, Any]) -> str:
    filename = _safe_video_filename(scene)
    video_id = filename.stem
    if len(filename.parts) != 1 or filename.name != f'{video_id}.mp4':
        raise ValueError(f'video filename must match official video_#####.mp4 layout: {filename!s}')
    if VIDEO_ID_RE.fullmatch(video_id) is None:
        raise ValueError(f'video filename must match official video_#####.mp4 layout: {video_id!r}')
    return video_id


def _validate_scene_identity(scene: Mapping[str, Any], *, video_id: str, source_role: str) -> None:
    scene_index = scene.get('scene_index')
    if not isinstance(scene_index, int) or isinstance(scene_index, bool):
        raise ValueError(f'{video_id}: scene_index must be an integer')
    filename_index = int(VIDEO_ID_RE.fullmatch(video_id).group(1))
    if scene_index != filename_index:
        raise ValueError(f'{video_id}: scene_index {scene_index} does not match filename index {filename_index}')
    lower, upper = (0, 10000) if source_role == 'official_train' else (10000, 15000)
    if not lower <= scene_index < upper:
        raise ValueError(f'{video_id}: scene_index is outside the {source_role} range [{lower}, {upper})')


def _program(value: Any, *, context: str) -> List[str]:
    program = canonical_program(value)
    if program is None:
        raise ValueError(
            f'{context}: program must match the official non-empty list[str] postfix-token schema')
    return program


def _question_reference(question: Mapping[str, Any], *, context: str) -> Dict[str, Any]:
    raw_type = question.get('question_type')
    if not isinstance(raw_type, str) or raw_type.casefold().strip() not in QUESTION_TYPES:
        raise ValueError(f'{context}: question_type must be one of {QUESTION_TYPES}')
    question_type = raw_type.casefold().strip()
    question_subtype = question.get('question_subtype')
    if question_type == 'descriptive':
        if question_subtype not in DESCRIPTIVE_SUBTYPES:
            raise ValueError(f'{context}: invalid or missing descriptive question_subtype')
    elif question_subtype is not None:
        raise ValueError(f'{context}: multiple-choice question must not define question_subtype')
    program = _program(question.get('program'), context=context)
    choices = question.get('choices')
    display_choices: List[Dict[str, Any]] = []

    if choices is None:
        answer = question.get('answer')
        if answer is None or not str(answer).strip():
            raise ValueError(f'{context}: missing descriptive answer')
        reference_answer = str(answer).strip()
        reward_metadata = {
            'answer_kind': 'short_text',
            'correct_choice_ids': [],
            'selected_choice_programs': [],
        }
    else:
        if not isinstance(choices, list) or not 1 <= len(choices) <= 4:
            raise ValueError(f'{context}: choices must be a list containing 1 to 4 entries')
        correct_ids: List[int] = []
        selected_programs: List[Dict[str, Any]] = []
        seen_ids = set()
        for offset, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                raise ValueError(f'{context}: each choice must be an object')
            choice_id = choice.get('choice_id')
            if not isinstance(choice_id, int) or isinstance(choice_id, bool) or choice_id != offset:
                raise ValueError(f'{context}: choice_id must be explicit and 0-based sequential')
            if choice_id in seen_ids:
                raise ValueError(f'{context}: choice_id must be unique')
            seen_ids.add(choice_id)
            choice_text = choice.get('choice')
            if not isinstance(choice_text, str) or not choice_text.strip():
                raise ValueError(f'{context}/choice_{choice_id}: choice text must be non-empty')
            choice_program = _program(choice.get('program'), context=f'{context}/choice_{choice_id}')
            label = choice.get('answer')
            if not isinstance(label, str) or label.casefold().strip() not in {'correct', 'wrong'}:
                raise ValueError(f'{context}/choice_{choice_id}: answer label must be correct or wrong')
            display_choices.append({'choice_id': choice_id, 'choice': choice_text.strip()})
            if label.casefold().strip() == 'correct':
                correct_ids.append(choice_id)
                selected_programs.append({'choice_id': choice_id, 'program': choice_program})
        correct_ids.sort()
        selected_programs.sort(key=lambda item: item['choice_id'])
        display_choices.sort(key=lambda item: item['choice_id'])
        reference_answer = json.dumps(correct_ids, separators=(',', ':'))
        reward_metadata = {
            'answer_kind': 'choice_ids',
            'correct_choice_ids': correct_ids,
            'selected_choice_programs': selected_programs,
        }

    trace = build_reference_trace(
        question_type=question_type,
        program=program,
        reward_metadata=reward_metadata,
    )
    if trace is None:
        raise ValueError(f'{context}: cannot build a valid reference trace')
    return {
        'question_type': question_type,
        'question_subtype': question_subtype,
        'program': program,
        'reward_metadata': reward_metadata,
        'reference_answer': reference_answer,
        'solution': render_output(trace, reference_answer),
        'display_choices': display_choices,
    }


def _format_question(question_text: str, choices: Sequence[Mapping[str, Any]], media_tokens: Sequence[str]) -> str:
    lines = [*media_tokens, f'Question: {question_text.strip()}']
    if choices:
        lines.append('Choices:')
        for choice in choices:
            lines.append(f"[{choice['choice_id']}] {choice['choice']}")
        lines.append('In <answer>, return a sorted JSON list of correct choice IDs (for example, [0,2] or []).')
    else:
        lines.append('In <answer>, return only the short descriptive answer.')
    lines.append('Infer the public symbolic program and use the exact required tags.')
    return '\n'.join(lines)


def _ensure_under_root(root: Path, candidate: Path, *, context: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f'{context}: resolved path escapes its declared root: {candidate}') from error
    return candidate


def _media_payload(*, media_mode: str, video_path: Path, frames_root: Path,
                   video_id: str) -> Tuple[str, List[str], List[str]]:
    if media_mode == 'video':
        return 'videos', [str(video_path)], ['<video>']
    frame_count = int(media_mode.removeprefix('frames'))
    frame_paths = planned_frame_paths(frames_root, video_id, frame_count)
    return 'images', frame_paths, ['<image>'] * frame_count


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
            count += 1
    return count


def _rank_key(row: Mapping[str, Any], *, seed: int) -> Tuple[str, str, str]:
    identity = f"{row['video_id']}:{row['question_id']}"
    digest = hashlib.sha256(f'cap:{seed}:{identity}'.encode('utf-8')).hexdigest()
    return digest, str(row['video_id']), str(row['question_id'])


def _select_caps(candidates: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]], *, caps: SplitCaps,
                 seed: int) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate[0], candidate[1]['question_type'])].append(candidate)

    selected = []
    cap_values = asdict(caps)
    for split in SPLIT_NAMES:
        for question_type in QUESTION_TYPES:
            group = sorted(grouped[(split, question_type)], key=lambda item: _rank_key(item[1], seed=seed))
            selected.extend(group[:cap_values[split]])
    return sorted(selected, key=lambda item: (item[0], str(item[1]['video_id']), str(item[1]['question_id'])))


def convert_dataset(
    *,
    train_annotation_paths: Sequence[Path],
    validation_annotation_paths: Sequence[Path],
    video_root: Path,
    output_dir: Path,
    frames_root: Optional[Path] = None,
    train_video_template: str = 'train/{video_filename}',
    validation_video_template: str = 'validation/{video_filename}',
    media_mode: str = 'frames8',
    seed: int = 42,
    dev_ratio: float = 0.1,
    caps: SplitCaps = SplitCaps(),
    require_full_caps: bool = False,
    require_media: bool = True,
    max_completion_chars: int = 4096,
) -> Dict[str, Any]:
    """Convert official annotations without crossing video or source boundaries."""

    if media_mode not in {'video', 'frames4', 'frames8'}:
        raise ValueError('media_mode must be video, frames4, or frames8')
    if not isinstance(max_completion_chars, int) or isinstance(max_completion_chars, bool) or max_completion_chars <= 0:
        raise ValueError('max_completion_chars must be a positive integer')
    caps.validate()
    train_paths = [Path(path).resolve() for path in train_annotation_paths]
    validation_paths = [Path(path).resolve() for path in validation_annotation_paths]
    if not train_paths or not validation_paths:
        raise ValueError('both official train and official validation annotations are required')
    all_paths = train_paths + validation_paths
    if len(set(all_paths)) != len(all_paths):
        raise ValueError('annotation paths must be unique and cannot appear in both source roles')

    video_root = Path(video_root).resolve()
    output_dir = Path(output_dir).resolve()
    frames_root = Path(frames_root or video_root / 'sampled_frames').resolve()
    templates = {
        'official_train': train_video_template,
        'official_validation': validation_video_template,
    }
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output directory: {output_dir}')

    manifests: Dict[str, Dict[str, Any]] = {}
    seen_questions = set()
    candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    for annotation_path, source_role, scene in iter_annotation_scenes(train_paths, validation_paths):
        video_id = _canonical_video_id(scene)
        _validate_scene_identity(scene, video_id=video_id, source_role=source_role)
        if video_id in manifests:
            raise ValueError(f'duplicate video_id across annotation scenes or source roles: {video_id}')
        filename = _safe_video_filename(scene)
        scene_index = scene.get('scene_index')
        template_values = {
            'source_role': source_role,
            'video_filename': filename.as_posix(),
            'video_id': video_id,
            'scene_index': scene_index,
        }
        relative_video = Path(templates[source_role].format(**template_values))
        if relative_video.is_absolute():
            raise ValueError(f'{video_id}: video template must resolve relative to --video-root')
        video_path = _ensure_under_root(video_root, video_root / relative_video, context=video_id)
        split = assign_split(video_id, source_role=source_role, seed=seed, dev_ratio=dev_ratio)
        media_key, media_values, media_tokens = _media_payload(
            media_mode=media_mode,
            video_path=video_path,
            frames_root=frames_root,
            video_id=video_id,
        )
        questions = scene.get('questions')
        if not isinstance(questions, list) or not questions:
            raise ValueError(f'{annotation_path}: {video_id} questions must be a non-empty list')
        manifests[video_id] = {
            'video_id': video_id,
            'scene_index': scene_index,
            'source_role': source_role,
            'source_annotation': str(annotation_path),
            'video_filename': filename.as_posix(),
            'video_path': str(video_path),
            'split': split,
            'frames_4': planned_frame_paths(frames_root, video_id, 4),
            'frames_8': planned_frame_paths(frames_root, video_id, 8),
            'question_count': 0,
        }

        for offset, question in enumerate(questions):
            if not isinstance(question, Mapping):
                raise ValueError(f'{annotation_path}: {video_id} question {offset} must be an object')
            question_id = question.get('question_id')
            if isinstance(question_id, bool) or not isinstance(question_id, int) or question_id < 0:
                raise ValueError(f'{video_id}: question_id must be an explicit non-negative integer')
            unique_key = (video_id, str(question_id))
            if unique_key in seen_questions:
                raise ValueError(f'duplicate question key: {unique_key}')
            seen_questions.add(unique_key)
            context = f'{annotation_path}:{video_id}/question_{question_id}'
            reference = _question_reference(question, context=context)
            question_text = question.get('question')
            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError(f'{context}: question text must be non-empty')
            prompt_messages = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': _format_question(question_text, reference['display_choices'], media_tokens),
                },
            ]
            base = {
                'video_id': video_id,
                'question_id': question_id,
                'question_type': reference['question_type'],
                'question_subtype': reference['question_subtype'],
                'program': reference['program'],
                'reward_metadata': reference['reward_metadata'],
                'solution': reference['solution'],
                'max_completion_chars': max_completion_chars,
                media_key: media_values,
            }
            grpo_row = dict(base, messages=prompt_messages)
            sft_row = dict(base, messages=prompt_messages + [{'role': 'assistant', 'content': reference['solution']}])
            candidates.append((split, sft_row, grpo_row))

    selected = _select_caps(candidates, caps=caps, seed=seed)
    if require_media:
        checked_media = set()
        for _, row, _ in selected:
            media_key = 'images' if 'images' in row else 'videos'
            for media_path in row[media_key]:
                if media_path in checked_media:
                    continue
                checked_media.add(media_path)
                if not Path(media_path).is_file():
                    raise FileNotFoundError(
                        f'{row["video_id"]}: selected {media_key} file is missing: {media_path}')
    selected_question_counts = Counter((split, sft['question_type']) for split, sft, _ in selected)
    cap_values = asdict(caps)
    underfilled = {
        f'{split}/{question_type}': {
            'actual': selected_question_counts[(split, question_type)],
            'requested': cap_values[split],
        }
        for split in SPLIT_NAMES
        for question_type in QUESTION_TYPES if selected_question_counts[(split, question_type)] < cap_values[split]
    }
    if require_full_caps and underfilled:
        raise ValueError(f'not enough labeled questions to fill requested caps: {underfilled}')

    selected_video_ids = {sft['video_id'] for _, sft, _ in selected}
    selected_manifests = [manifests[video_id] for video_id in sorted(selected_video_ids)]
    for manifest in selected_manifests:
        manifest['question_count'] = sum(1 for split, sft, _ in selected
                                         if split == manifest['split'] and sft['video_id'] == manifest['video_id'])

    video_sets = {
        split: {row['video_id']
                for row in selected_manifests if row['split'] == split}
        for split in SPLIT_NAMES
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            overlap = video_sets[left] & video_sets[right]
            if overlap:
                raise AssertionError(f'video leakage between {left} and {right}: {sorted(overlap)[:3]}')
    if any(manifests[video_id]['source_role'] != 'official_validation' for video_id in video_sets['test']):
        raise AssertionError('test contains a video outside official validation')
    if any(manifests[video_id]['source_role'] != 'official_train'
           for video_id in video_sets['train'] | video_sets['dev']):
        raise AssertionError('train/dev contains a video outside official train')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f'.{output_dir.name}.staging-{uuid.uuid4().hex}'
    stage.mkdir()
    try:
        output_files: Dict[str, Dict[str, Any]] = {}
        manifest_path = stage / 'manifest.jsonl'
        output_files[manifest_path.name] = {'rows': _write_jsonl(manifest_path, selected_manifests)}
        for split in SPLIT_NAMES:
            sft_path = stage / f'sft_{split}.jsonl'
            grpo_path = stage / f'grpo_{split}.jsonl'
            output_files[sft_path.name] = {
                'rows': _write_jsonl(sft_path, (sft for name, sft, _ in selected if name == split))
            }
            output_files[grpo_path.name] = {
                'rows': _write_jsonl(grpo_path, (grpo for name, _, grpo in selected if name == split))
            }
        for filename, metadata in output_files.items():
            metadata['sha256'] = _sha256(stage / filename)

        summary = {
            'schema_version':
            1,
            'seed':
            seed,
            'dev_ratio':
            dev_ratio,
            'caps_per_question_type':
            cap_values,
            'media_mode':
            media_mode,
            'annotation_files': [{
                'path': str(path),
                'source_role': role,
                'sha256': _sha256(path)
            } for role, paths in (('official_train', train_paths), ('official_validation', validation_paths))
                                 for path in paths],
            'output_dir':
            str(output_dir),
            'frames_root':
            str(frames_root),
            'video_counts': {
                name: len(video_sets[name])
                for name in SPLIT_NAMES
            },
            'question_counts': {
                split: {
                    question_type: selected_question_counts[(split, question_type)]
                    for question_type in QUESTION_TYPES
                }
                for split in SPLIT_NAMES
            },
            'underfilled_caps':
            underfilled,
            'video_disjoint':
            True,
            'source_protocol': {
                'train': 'official_train',
                'dev': 'official_train_hash_holdout',
                'test': 'official_validation_only',
            },
            'files':
            output_files,
        }
        with (stage / 'split_summary.json').open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(stage, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-annotations', type=Path, nargs='+', required=True)
    parser.add_argument('--validation-annotations', type=Path, nargs='+', required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames-root', type=Path)
    parser.add_argument('--train-video-template', default='train/{video_filename}')
    parser.add_argument('--validation-video-template', default='validation/{video_filename}')
    parser.add_argument('--media-mode', choices=('video', 'frames4', 'frames8'), default='frames8')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--train-per-type', type=int, default=3000)
    parser.add_argument('--dev-per-type', type=int, default=500)
    parser.add_argument('--test-per-type', type=int, default=1000)
    parser.add_argument('--require-full-caps', action='store_true')
    parser.add_argument('--allow-missing-media', action='store_true')
    parser.add_argument('--max-completion-chars', type=int, default=4096)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = convert_dataset(
        train_annotation_paths=args.train_annotations,
        validation_annotation_paths=args.validation_annotations,
        video_root=args.video_root,
        output_dir=args.output_dir,
        frames_root=args.frames_root,
        train_video_template=args.train_video_template,
        validation_video_template=args.validation_video_template,
        media_mode=args.media_mode,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        caps=SplitCaps(args.train_per_type, args.dev_per_type, args.test_per_type),
        require_full_caps=args.require_full_caps,
        require_media=not args.allow_missing_media,
        max_completion_chars=args.max_completion_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
