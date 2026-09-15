"""Small dependency-free identity and JSON helpers."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(row):
    if not isinstance(row.get('video_id'), str) or type(row.get('question_id')) is not int:
        raise ValueError('video_id must be string and question_id must be integer')
    if row['question_id'] < 0:
        raise ValueError('question_id must not be negative')
    return hashlib.sha256(f"{row['video_id']}:{row['question_id']}".encode('utf-8')).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('JSONL must contain at least one object')
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def unique_rows(rows):
    result = {}
    for row in rows:
        identity = record_id(row)
        if identity in result:
            raise ValueError(f'duplicate video/question identity: {identity}')
        result[identity] = row
    return result
