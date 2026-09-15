"""Single-adapter offline inference. No weights, CUDA or API are touched in dry-run mode."""

import argparse
import json
import os
from pathlib import Path
import time

from .common import read_jsonl, record_id, sha256, unique_rows, write_json
from .runtime import (add_gpu_arguments, check_versions, gpu_lease, prepare_environment,
                      require_local_model, validate_execute_arguments)


def public_payload(row):
    messages = row.get('messages', [])
    if [message.get('role') for message in messages] != ['system', 'user']:
        raise ValueError('Use grpo_dev/grpo_test rows: only system/user messages may reach inference')
    if not isinstance(row.get('images'), list) or len(row['images']) != 8:
        raise ValueError('Fixed protocol requires exactly eight frames')
    # Explicit allowlist: solution, program, reward_metadata and every extra label stay outside the model.
    return {'messages': [{'role': item['role'], 'content': item['content']} for item in messages],
            'images': list(row['images'])}


def argument_values(model, adapter=None):
    return {'model': str(model), 'adapters': [] if adapter is None else [str(adapter)],
            'load_args': False, 'remove_unused_columns': True, 'template': 'qwen3_vl',
            'infer_backend': 'transformers', 'attn_impl': 'sdpa', 'torch_dtype': 'bfloat16',
            'max_length': 2048, 'truncation_strategy': 'delete', 'max_pixels': 50176,
            'max_batch_size': 1, 'stream': False, 'seed': 42}


def decode_values():
    return {'max_tokens': 320, 'temperature': 0, 'top_k': 50, 'top_p': 1.0,
            'repetition_penalty': 1.0, 'num_beams': 1, 'stream': False, 'n': 1, 'return_details': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, help='New prediction JSONL path')
    parser.add_argument('--adapter', help='Omit for Base; supply one SFT or GRPO adapter, never stack the two')
    add_gpu_arguments(parser)
    args = parser.parse_args()
    plan = {'arguments': argument_values(args.model, args.adapter), 'decode': decode_values(),
            'dataset': args.dataset, 'output': args.output,
            'model_input_fields': ['messages (system/user only)', 'images'],
            'mode': 'execute' if args.execute else 'dry_run'}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    validate_execute_arguments(args)
    require_local_model(args.model)
    if args.adapter and not (Path(args.adapter).is_absolute() and (Path(args.adapter) / 'adapter_model.safetensors').is_file()):
        raise ValueError('--adapter must be an existing absolute local adapter directory')
    rows = read_jsonl(args.dataset)
    unique_rows(rows)
    for row in rows:
        payload = public_payload(row)
        if any(not Path(image).is_file() for image in payload['images']):
            raise FileNotFoundError('An input frame is missing')
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or Path(str(output) + '.run.json').exists():
        raise ValueError('Output must be a new absolute prediction path on your data disk')
    versions = check_versions()
    environment = prepare_environment(args.cache_root, args.expected_gpu_uuid)
    with gpu_lease(args.cache_root, args.gpu, args.expected_gpu_uuid) as gpu:
        os.environ.update(environment)
        import torch
        from swift.arguments import InferArguments
        from swift.infer_engine import InferRequest, RequestConfig
        from swift.pipelines.infer.infer import SwiftInfer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one selected CUDA device is required')
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(output) + '.run.json', {**plan, 'dataset_sha256': sha256(args.dataset),
                                             'versions': versions, 'gpu': gpu, 'status': 'started'})
        runner = SwiftInfer(InferArguments(**plan['arguments']))
        generated = 0
        try:
            with output.open('x', encoding='utf-8', newline='\n') as handle:
                for ordinal, row in enumerate(rows):
                    response = runner.infer_engine.infer(
                        [InferRequest(**public_payload(row))], request_config=RequestConfig(**plan['decode']),
                        use_tqdm=False,
                    )[0]
                    if isinstance(response, BaseException):
                        raise response
                    choice = response.choices[0]
                    if not isinstance(choice.message.content, str):
                        raise ValueError('Inference returned non-text content')
                    prediction = {'ordinal': ordinal, 'record_id': record_id(row), 'video_id': row['video_id'],
                                  'question_id': row['question_id'], 'question_type': row['question_type'],
                                  'response': choice.message.content, 'finish_reason': choice.finish_reason,
                                  'prompt_tokens': response.usage.prompt_tokens,
                                  'completion_tokens': response.usage.completion_tokens}
                    handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                    handle.flush()
                    generated += 1
        finally:
            write_json(str(output) + '.exit.json', {'status': 'completed' if generated == len(rows) else 'incomplete',
                                                   'generated': generated, 'expected': len(rows),
                                                   'wall_seconds': time.monotonic() - started,
                                                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                                   'note': 'No failed/malformed answer retries. Partial output is retained, never silently scored.'})


if __name__ == '__main__':
    main()
