"""Opt-in GPU guard. No GPU libraries are imported during dry runs."""

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

CORE_VERSIONS = {'torch': '2.6.0+cu124', 'transformers': '4.57.3', 'ms-swift': '4.5.2',
                 'peft': '0.19.1', 'trl': '0.29.1'}


def check_versions():
    installed = {}
    for package, expected in CORE_VERSIONS.items():
        actual = importlib.metadata.version(package)
        installed[package] = actual
        if actual != expected:
            raise RuntimeError(f'{package}: expected recorded version {expected}; found {actual}')
    if sys.version_info[:2] != (3, 11) or sys.platform != 'linux':
        raise RuntimeError('Recorded GPU recipe requires Linux and Python 3.11; CPU utilities are portable.')
    return installed


def prepare_environment(cache_root, uuid):
    root = Path(cache_root)
    if not root.is_absolute():
        raise ValueError('--cache-root must explicitly name an absolute directory on your data disk')
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': uuid, 'MAX_PIXELS': '50176', 'WANDB_DISABLED': 'true',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MODELSCOPE_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false'})
    for name, subdir in {
        'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': 'huggingface/hub', 'HF_ASSETS_CACHE': 'huggingface/assets',
        'HF_XET_CACHE': 'huggingface/xet', 'TRANSFORMERS_CACHE': 'huggingface/transformers',
        'HF_DATASETS_CACHE': 'huggingface/datasets', 'MODELSCOPE_CACHE': 'modelscope',
        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
        'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg',
        'PIP_CACHE_DIR': 'pip', 'TMPDIR': 'tmp', 'PYTHONPYCACHEPREFIX': 'pycache',
        'TORCHINDUCTOR_CACHE_DIR': 'torchinductor', 'VLLM_CACHE_ROOT': 'vllm', 'NUMBA_CACHE_DIR': 'numba',
    }.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'NPROC_PER_NODE', 'NNODES', 'NODE_RANK',
                'MASTER_ADDR', 'MASTER_PORT', '_PATCH_WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'LOCAL_SIZE'):
        env.pop(key, None)
    return env


def inspect_gpu(index, expected_uuid):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    devices = [line.split(',') for line in result.stdout.strip().splitlines()]
    selected = [parts for parts in devices if parts[0].strip() == str(index)]
    if len(selected) != 1:
        raise RuntimeError('Selected physical GPU index is not uniquely visible')
    _, uuid, memory = [value.strip() for value in selected[0]]
    if uuid != expected_uuid or int(memory) >= 500:
        raise RuntimeError('GPU UUID changed or memory.used >= 500 MiB; refusing launch')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.stdout.splitlines()):
        raise RuntimeError('The selected GPU already has a compute process')
    return {'index': index, 'uuid': uuid, 'memory_used_mib': int(memory)}


@contextlib.contextmanager
def gpu_lease(cache_root, index, expected_uuid):
    # This cooperative file lock protects only launchers using the same cache root.
    # It cannot reserve a shared server GPU against unrelated schedulers/users.
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f'framereason-gpu-{index}.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump({'pid': os.getpid(), 'uuid': expected_uuid}, handle)
        yield inspect_gpu(index, expected_uuid)
    finally:
        # Only our successfully created lock is removed; stale foreign locks are never reclaimed automatically.
        lock.unlink(missing_ok=True)


def add_gpu_arguments(parser):
    parser.add_argument('--execute', action='store_true', help='Explicitly execute; default is a CPU-only dry run')
    parser.add_argument('--gpu', type=int, help='Authorized physical GPU index')
    parser.add_argument('--expected-gpu-uuid', help='UUID checked against nvidia-smi immediately before launch')
    parser.add_argument('--cache-root', help='Absolute cache directory on your data disk')


def validate_execute_arguments(args):
    if args.gpu is None or not args.expected_gpu_uuid or not args.cache_root:
        raise ValueError('--execute requires --gpu, --expected-gpu-uuid and --cache-root')
    if args.gpu < 0 or not Path(args.cache_root).is_absolute():
        raise ValueError('GPU index must be nonnegative and cache root must be absolute')


def require_local_model(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('--model must be an existing absolute local model snapshot, not a download identifier')
    return path
