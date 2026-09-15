"""Uniformly extract 4/8 CLEVRER frames from a converter manifest with ffmpeg."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_non_finite(value):
    raise ValueError(f'non-finite JSON value: {value}')


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_indices(total_frames: int, output_frames: int) -> List[int]:
    """Return deterministic endpoint-inclusive indices with no duplicates."""

    if not isinstance(total_frames, int) or isinstance(total_frames, bool) or total_frames <= 0:
        raise ValueError('total_frames must be a positive integer')
    if not isinstance(output_frames, int) or isinstance(output_frames, bool) or output_frames <= 0:
        raise ValueError('output_frames must be a positive integer')
    if total_frames < output_frames:
        raise ValueError(f'video has {total_frames} frames but {output_frames} are required')
    if output_frames == 1:
        return [(total_frames - 1) // 2]
    indices = [round(index * (total_frames - 1) / (output_frames - 1)) for index in range(output_frames)]
    if len(set(indices)) != output_frames:
        raise AssertionError('uniform frame selection produced duplicate indices')
    return indices


def _reject_symlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f'symlink path component is not allowed: {current}')


def _frozen_frames_root(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('frames_root must be an explicit absolute path')
    frozen = Path(os.path.abspath(path))
    _reject_symlink_chain(frozen)
    if not frozen.is_dir():
        raise FileNotFoundError(f'frames_root must already exist as a real directory: {frozen}')
    return frozen


def _load_manifest(path: Path, *, frame_count: int, frames_root: Path) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _strict_json_loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f'{path}:{line_number}: row must be an object')
            video_id = row.get('video_id')
            video_path = row.get('video_path')
            frame_paths = row.get(f'frames_{frame_count}')
            if not isinstance(video_id, str) or not video_id or video_id in seen:
                raise ValueError(f'{path}:{line_number}: video_id must be unique and non-empty')
            if not isinstance(video_path, str) or not video_path:
                raise ValueError(f'{path}:{line_number}: video_path must be non-empty')
            if not isinstance(frame_paths, list) or len(frame_paths) != frame_count:
                raise ValueError(f'{path}:{line_number}: frames_{frame_count} must contain exactly {frame_count} paths')
            if not all(isinstance(item, str) and Path(item).is_absolute() for item in frame_paths):
                raise ValueError(f'{path}:{line_number}: frame paths must be absolute strings')
            targets = [Path(os.path.abspath(item)) for item in frame_paths]
            expected_dir = frames_root / f'{frame_count}f' / video_id
            expected_names = [f'frame_{index:03d}.jpg' for index in range(frame_count)]
            expected_targets = [expected_dir / name for name in expected_names]
            if targets != expected_targets:
                raise ValueError(
                    f'{path}:{line_number}: frame paths escape frames_root or violate the canonical layout')
            _reject_symlink_chain(expected_dir)
            seen.add(video_id)
            rows.append({'video_id': video_id, 'video_path': str(Path(video_path).resolve()), 'targets': targets})
    if not rows:
        raise ValueError(f'{path}: manifest is empty')
    return rows


def _program_version(executable: str) -> str:
    result = subprocess.run(
        [executable, '-version'],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = (result.stdout or result.stderr).splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f'{executable}: version command returned no usable output')
    return lines[0].strip()


def _resolved_program(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f'executable is unavailable: {executable}')
        resolved = str(candidate)
    return str(Path(resolved).resolve())


def _imageio_ffmpeg():
    try:
        return importlib.import_module('imageio_ffmpeg')
    except ImportError as error:
        raise RuntimeError(
            'ffprobe is unavailable and the required imageio_ffmpeg fallback is not installed') from error


def _positive_frame_count(value: Any, *, backend: str, allow_digit_string: bool = False) -> int:
    if allow_digit_string and isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or value <= 0:
        raise ValueError(f'{backend} returned an invalid frame count: {value!r}')
    return value


def _imageio_frame_count(video_path: Path) -> tuple[int, Dict[str, str]]:
    imageio_ffmpeg = _imageio_ffmpeg()
    count, _ = imageio_ffmpeg.count_frames_and_secs(video_path)
    count = _positive_frame_count(count, backend='imageio_ffmpeg.count_frames_and_secs')
    executable = _resolved_program(imageio_ffmpeg.get_ffmpeg_exe())
    return count, {
        'backend': 'imageio_ffmpeg.count_frames_and_secs',
        'backend_version': str(imageio_ffmpeg.__version__),
        'executable': executable,
        'executable_version': _program_version(executable),
    }


def _probe_frame_count(video_path: Path, *, ffprobe: str) -> tuple[int, Dict[str, str]]:
    command = [
        ffprobe,
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-count_frames',
        '-show_entries',
        'stream=nb_read_frames,nb_frames',
        '-of',
        'json',
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return _imageio_frame_count(video_path)
    payload = _strict_json_loads(result.stdout)
    streams = payload.get('streams') if isinstance(payload, Mapping) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise ValueError(f'{video_path}: ffprobe returned no unique video stream')
    for key in ('nb_read_frames', 'nb_frames'):
        value = streams[0].get(key)
        try:
            count = _positive_frame_count(value, backend='ffprobe', allow_digit_string=True)
        except ValueError:
            continue
        executable = _resolved_program(ffprobe)
        return count, {
            'backend': 'ffprobe',
            'executable': executable,
            'executable_version': _program_version(executable),
        }
    raise ValueError(f'{video_path}: ffprobe did not return a usable frame count')


def _resolve_ffmpeg(ffmpeg: str) -> Dict[str, str]:
    if ffmpeg == 'ffmpeg':
        try:
            executable = _resolved_program(_imageio_ffmpeg().get_ffmpeg_exe())
            backend = 'imageio_ffmpeg'
        except (FileNotFoundError, RuntimeError):
            executable = _resolved_program(ffmpeg)
            backend = 'system'
    else:
        executable = _resolved_program(ffmpeg)
        backend = 'explicit'
    return {
        'backend': backend,
        'executable': executable,
        'executable_version': _program_version(executable),
        'output_sync': '-vsync vfr',
    }


def _existing_complete(
    targets: Sequence[Path], *, video_id: str, video_path: Path, video_sha256: str, frame_count: int, frames_root: Path
) -> Optional[Dict[str, Any]]:
    target_dir = targets[0].parent
    if not target_dir.exists():
        return None
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError(f'frame target must be a real directory: {target_dir}')
    existing_names = sorted(path.name for path in target_dir.iterdir())
    expected_names = sorted([*(path.name for path in targets), '.extract.json'])
    if (existing_names != expected_names
            or not all(not path.is_symlink() and path.is_file() and path.stat().st_size > 0 for path in targets)):
        raise FileExistsError(f'partial or non-canonical frame directory: {target_dir}')
    sidecar_path = target_dir / '.extract.json'
    if sidecar_path.is_symlink():
        raise ValueError(f'extraction sidecar must not be a symlink: {sidecar_path}')
    try:
        sidecar = _strict_json_loads(sidecar_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f'invalid extraction sidecar: {sidecar_path}') from error
    if not isinstance(sidecar, Mapping):
        raise ValueError(f'invalid extraction sidecar: {sidecar_path}')
    expected_core = {
        'video_id': video_id,
        'video_path': str(video_path),
        'video_sha256': video_sha256,
        'frame_count': frame_count,
        'frames_root': str(frames_root),
    }
    if any(sidecar.get(key) != value for key, value in expected_core.items()):
        raise ValueError(f'extraction sidecar/source mismatch: {sidecar_path}')
    indices = sidecar.get('selected_indices')
    total_frames = sidecar.get('total_frames')
    if not isinstance(total_frames, int) or indices != uniform_indices(total_frames, frame_count):
        raise ValueError(f'extraction sidecar has invalid frame indices: {sidecar_path}')
    frame_records = sidecar.get('frames')
    if not isinstance(frame_records, list) or len(frame_records) != frame_count:
        raise ValueError(f'extraction sidecar has invalid frame records: {sidecar_path}')
    for target, record in zip(targets, frame_records):
        if (not isinstance(record, Mapping) or record.get('path') != str(target)
                or record.get('sha256') != _sha256(target)):
            raise ValueError(f'extraction sidecar frame hash mismatch: {target}')
    return dict(sidecar)


def extract_frames(
    *,
    manifest_path: Path,
    frames_root: Path,
    receipt_path: Path,
    frame_count: int,
    ffmpeg: str = 'ffmpeg',
    ffprobe: str = 'ffprobe',
) -> Dict[str, Any]:
    """Extract every manifest video atomically and write a hash receipt."""

    if frame_count not in {4, 8}:
        raise ValueError('frame_count must be 4 or 8')
    manifest_path = Path(manifest_path).resolve()
    frames_root = _frozen_frames_root(frames_root)
    receipt_path = Path(receipt_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if receipt_path.exists():
        raise FileExistsError(f'refusing to overwrite receipt: {receipt_path}')
    rows = _load_manifest(manifest_path, frame_count=frame_count, frames_root=frames_root)
    receipts = []
    generated = 0
    reused = 0

    for row in rows:
        video_path = Path(row['video_path'])
        targets = row['targets']
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_sha256 = _sha256(video_path)
        sidecar = _existing_complete(
            targets,
            video_id=row['video_id'],
            video_path=video_path,
            video_sha256=video_sha256,
            frame_count=frame_count,
            frames_root=frames_root,
        )
        if sidecar is not None:
            reused += 1
        else:
            total_frames, frame_count_probe = _probe_frame_count(video_path, ffprobe=ffprobe)
            indices = uniform_indices(total_frames, frame_count)
            ffmpeg_provenance = _resolve_ffmpeg(ffmpeg)
            target_dir = targets[0].parent
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            stage = target_dir.parent / f'.{target_dir.name}.staging-{uuid.uuid4().hex}'
            stage.mkdir()
            try:
                expression = '+'.join(f'eq(n\\,{index})' for index in indices)
                subprocess.run(
                    [
                        ffmpeg_provenance['executable'],
                        '-hide_banner',
                        '-loglevel',
                        'error',
                        '-nostdin',
                        '-i',
                        str(video_path),
                        '-vf',
                        f'select={expression}',
                        '-vsync',
                        'vfr',
                        '-start_number',
                        '0',
                        '-frames:v',
                        str(frame_count),
                        str(stage / 'frame_%03d.jpg'),
                    ],
                    check=True,
                )
                staged = [stage / target.name for target in targets]
                if not all(path.is_file() and path.stat().st_size > 0 for path in staged):
                    raise RuntimeError(
                        f'{row["video_id"]}: ffmpeg did not produce exactly {frame_count} non-empty frames')
                sidecar = {
                    'schema_version': 1,
                    'video_id': row['video_id'],
                    'video_path': str(video_path),
                    'video_sha256': video_sha256,
                    'total_frames': total_frames,
                    'frame_count': frame_count,
                    'frames_root': str(frames_root),
                    'selected_indices': indices,
                    'selection_filter': expression,
                    'frame_count_probe': frame_count_probe,
                    'ffmpeg': ffmpeg_provenance,
                    'frames': [{
                        'path': str(target),
                        'sha256': _sha256(staged_path)
                    } for target, staged_path in zip(targets, staged)],
                }
                with (stage / '.extract.json').open('w', encoding='utf-8', newline='\n') as handle:
                    json.dump(sidecar, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                    handle.write('\n')
                os.replace(stage, target_dir)
                generated += 1
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        receipts.append({
            'video_id': row['video_id'],
            'video_path': str(video_path),
            'video_sha256': video_sha256,
            'selected_indices': sidecar['selected_indices'],
            'frame_count_probe': sidecar.get('frame_count_probe'),
            'ffmpeg': sidecar.get('ffmpeg'),
            'frames': [{
                'path': str(path),
                'sha256': _sha256(path)
            } for path in targets],
        })

    receipt = {
        'schema_version': 1,
        'source_manifest': {
            'path': str(manifest_path),
            'sha256': _sha256(manifest_path)
        },
        'frame_count': frame_count,
        'frames_root': str(frames_root),
        'video_count': len(rows),
        'generated_video_count': generated,
        'reused_video_count': reused,
        'videos': receipts,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.parent / f'.{receipt_path.name}.tmp-{uuid.uuid4().hex}'
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, receipt_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--frames-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--frame-count', type=int, choices=(4, 8), required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = extract_frames(
        manifest_path=args.manifest,
        frames_root=args.frames_root,
        receipt_path=args.receipt,
        frame_count=args.frame_count,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != 'videos'}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
