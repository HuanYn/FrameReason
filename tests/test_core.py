"""CPU-only tests. Synthetic fixtures are authored here, not model experiment outputs."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

from framereason.common import record_id, sha256
from framereason.convert_clevrer import SplitCaps, assign_split, convert_dataset
from framereason.evaluate import evaluate
from framereason.extract_frames import uniform_indices
from framereason.infer import argument_values, decode_values, public_payload
from framereason.launch import build_command
from framereason.output_format import parse_output, render_output
from framereason.reward_core import answer_is_correct, build_reference_trace, score_completion
from framereason.runtime import CORE_VERSIONS, check_versions, inspect_gpu, prepare_environment
from framereason.subset import TYPES, rank_key, select_subset


ROOT = Path(__file__).resolve().parents[1]


def fixture(question_id=0, question_type='descriptive'):
    metadata = {'answer_kind': 'short_text', 'correct_choice_ids': [], 'selected_choice_programs': []}
    answer = '2'
    if question_type != 'descriptive':
        metadata = {'answer_kind': 'choice_ids', 'correct_choice_ids': [1],
                    'selected_choice_programs': [{'choice_id': 1, 'program': ['events', 'filter_collision']}]}
        answer = '[1]'
    row = {'video_id': 'video_11000', 'question_id': question_id, 'question_type': question_type,
           'program': ['scene', 'count'], 'reward_metadata': metadata, 'max_completion_chars': 4096,
           'messages': [{'role': 'system', 'content': 'synthetic system prompt'},
                        {'role': 'user', 'content': '<image>' * 8 + 'Synthetic question'}],
           'images': [f'/synthetic/frame_{i:03}.jpg' for i in range(8)]}
    trace = build_reference_trace(question_type=question_type, program=row['program'], reward_metadata=metadata)
    row['solution'] = render_output(trace, answer)
    return row


def prediction(row, response=None):
    return {'record_id': record_id(row), 'video_id': row['video_id'], 'question_id': row['question_id'],
            'question_type': row['question_type'], 'response': row['solution'] if response is None else response}


class OutputAndRewardTests(unittest.TestCase):
    def test_exact_reference_gets_one(self):
        row = fixture()
        score = score_completion(row['solution'], reference_answer='2', question_type=row['question_type'],
                                 program=row['program'], reward_metadata=row['reward_metadata'])
        self.assertEqual(score.total, 1.0)
        self.assertTrue(score.answer_correct)
        self.assertTrue(score.trace_consistent)

    def test_invalid_json_keeps_answer_only(self):
        parsed = parse_output('<trace>{broken}</trace><answer>2</answer>')
        self.assertFalse(parsed.format_valid)
        self.assertEqual(parsed.answer, '2')

    def test_duplicate_and_nan_rejected(self):
        for trace in ('{"a":1,"a":2}', '{"a":NaN}'):
            self.assertFalse(parse_output(f'<trace>{trace}</trace><answer>2</answer>').format_valid)

    def test_ambiguous_answer_never_guessed(self):
        self.assertIsNone(parse_output('<trace>{}</trace><answer>2</answer><answer>3</answer>').answer)

    def test_choice_answer_exact_sorted_set(self):
        self.assertTrue(answer_is_correct('[0, 2]', '[0,2]', answer_kind='choice_ids'))
        for invalid in ('[2,0]', '[0,0,2]', '[true,2]', '[0,1,2]'):
            self.assertFalse(answer_is_correct(invalid, '[0,2]', answer_kind='choice_ids'))

    def test_short_answer_no_extra_description(self):
        self.assertTrue(answer_is_correct(' CYLINDER ', 'cylinder', answer_kind='short_text'))
        self.assertFalse(answer_is_correct('purple cylinder', 'cylinder', answer_kind='short_text'))

    def test_penalty_is_capped_once(self):
        row = fixture()
        repeated = row['solution'] + row['solution']
        score = score_completion(repeated, reference_answer='2', question_type=row['question_type'],
                                 program=row['program'], reward_metadata=row['reward_metadata'], max_completion_chars=1)
        self.assertEqual(score.penalty, 0.1)
        self.assertTrue(score.repeated and score.overlength)

    def test_trace_annotation_mismatch_not_physics(self):
        row = fixture()
        trace = parse_output(row['solution']).trace
        trace['question_program'] = ['scene', 'exist']
        score = score_completion(render_output(trace, '2'), reference_answer='2', question_type=row['question_type'],
                                 program=row['program'], reward_metadata=row['reward_metadata'])
        self.assertEqual(score.total, 0.6)
        self.assertFalse(score.trace_consistent)


class DataTests(unittest.TestCase):
    def test_endpoint_uniform_frames(self):
        self.assertEqual(uniform_indices(128, 8), [0, 18, 36, 54, 73, 91, 109, 127])
        with self.assertRaises(ValueError):
            uniform_indices(7, 8)

    def test_official_validation_never_training(self):
        self.assertEqual(assign_split('video_11000', source_role='official_validation', seed=42, dev_ratio=.1), 'test')

    def test_converter_roundtrip_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_id = next(i for i in range(100) if assign_split(f'video_{i:05}', source_role='official_train', seed=42, dev_ratio=.1) == 'train')
            dev_id = next(i for i in range(100) if assign_split(f'video_{i:05}', source_role='official_train', seed=42, dev_ratio=.1) == 'dev')

            def scene(index):
                return {'scene_index': index, 'video_filename': f'video_{index:05}.mp4', 'questions': [{
                    'question_id': 0, 'question_type': 'descriptive', 'question_subtype': 'count',
                    'question': 'How many synthetic objects are present?', 'answer': '2', 'program': ['scene', 'count']}]}

            train = root / 'train.json'
            validation = root / 'validation.json'
            train.write_text(json.dumps([scene(train_id), scene(dev_id)]), encoding='utf-8')
            validation.write_text(json.dumps([scene(11000)]), encoding='utf-8')
            output = root / 'converted'
            report = convert_dataset(train_annotation_paths=[train], validation_annotation_paths=[validation],
                                     video_root=root / 'videos', frames_root=root / 'frames', output_dir=output,
                                     caps=SplitCaps(1, 1, 1), require_media=False)
            self.assertTrue(report['video_disjoint'])
            for split in ('train', 'dev', 'test'):
                grpo = json.loads((output / f'grpo_{split}.jsonl').read_text(encoding='utf-8'))
                sft = json.loads((output / f'sft_{split}.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(len(grpo['images']), 8)
                self.assertEqual(len(grpo['messages']), 2)
                self.assertEqual(sft['messages'][-1]['content'], grpo['solution'])
                self.assertNotIn('solution', public_payload(grpo))
            with self.assertRaises(FileExistsError):
                convert_dataset(train_annotation_paths=[train], validation_annotation_paths=[validation],
                                video_root=root / 'videos', output_dir=output, require_media=False)

    def test_fixed800_hash_selection_is_answer_blind_and_order_independent(self):
        rows = [fixture(i, kind) for kind in TYPES for i in range(1000)]
        for type_index, kind in enumerate(TYPES):
            for row in rows[type_index * 1000:(type_index + 1) * 1000]:
                row['video_id'] = f'video_{11000 + type_index:05}'
        selected = select_subset(rows)
        altered = copy.deepcopy(rows)
        for row in altered:
            row['solution'] = 'Not used in selection'
        random.Random(18).shuffle(altered)
        self.assertEqual([record_id(row) for row in selected], [record_id(row) for row in select_subset(altered)])
        self.assertEqual(len(selected), 800)
        payload = ['clevrer.compact800.sampling/1', 42, rows[0]['question_type'], rows[0]['video_id'], rows[0]['question_id']]
        expected = hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(rank_key(rows[0])[0], expected)

    def test_published_core_source_hashes(self):
        provenance = json.loads((ROOT / 'configs/provenance.json').read_text(encoding='utf-8'))
        for filename, expected in provenance['published_lf_files_sha256'].items():
            self.assertEqual(sha256(ROOT / filename), expected, filename)


class EvaluationTests(unittest.TestCase):
    def test_correct_and_incorrect_pair(self):
        first, second = fixture(0), fixture(1, 'predictive')
        wrong = second['solution'].replace('<answer>[1]</answer>', '<answer>[]</answer>')
        report = evaluate([first, second], [prediction(second, wrong), prediction(first)])
        self.assertEqual(report['overall']['accuracy'], .5)
        self.assertFalse(report['event_executor'])

    def test_missing_duplicate_or_mismatched_identity_rejected(self):
        first, second = fixture(0), fixture(1)
        for predictions in ([prediction(first)], [prediction(first), prediction(first)]):
            with self.assertRaises(ValueError):
                evaluate([first, second], predictions)
        corrupted = prediction(first)
        corrupted['record_id'] = 'wrong'
        with self.assertRaises(ValueError):
            evaluate([first], [corrupted])

    def test_public_payload_drops_private_columns_and_rejects_sft_target(self):
        row = fixture()
        row['secret_label'] = 'must not reach model'
        self.assertEqual(set(public_payload(row)), {'messages', 'images'})
        row['messages'].append({'role': 'assistant', 'content': row['solution']})
        with self.assertRaises(ValueError):
            public_payload(row)


class RecipeTests(unittest.TestCase):
    def test_plugin_registration_and_batch_contract_without_swift(self):
        fake_swift = types.ModuleType('swift')
        fake_rewards = types.ModuleType('swift.rewards')
        fake_rewards.ORM = object
        fake_rewards.orms = {}
        with patch.dict(sys.modules, {'swift': fake_swift, 'swift.rewards': fake_rewards}):
            spec = importlib.util.spec_from_file_location('synthetic_reward_plugin', ROOT / 'framereason/reward_plugin.py')
            plugin = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin)
            scorer = fake_rewards.orms['clevrer_verifiable']()
            row = fixture()
            columns = {key: [row[key]] for key in ('solution', 'question_type', 'program', 'reward_metadata')}
            self.assertEqual(scorer([row['solution']], **columns), [1.0])
            with self.assertRaises(ValueError):
                scorer([row['solution']])

    def test_runtime_recorded_python311_and_core_versions(self):
        with patch('framereason.runtime.importlib.metadata.version', side_effect=lambda key: CORE_VERSIONS[key]), \
             patch('framereason.runtime.sys.version_info', (3, 11, 11)), \
             patch('framereason.runtime.sys.platform', 'linux'):
            self.assertEqual(check_versions(), CORE_VERSIONS)

    def test_environment_clears_multicard_launch_state(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'NPROC_PER_NODE': '8', 'WORLD_SIZE': '8', 'MASTER_ADDR': 'synthetic'}):
            environment = prepare_environment(directory, 'GPU-TEST')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-TEST')
            self.assertEqual(environment['MAX_PIXELS'], '50176')
            self.assertNotIn('NPROC_PER_NODE', environment)
            self.assertNotIn('WORLD_SIZE', environment)
            self.assertNotIn('MASTER_ADDR', environment)

    def test_historical_configs(self):
        for stage in ('sft', 'grpo'):
            self.assertEqual(sha256(ROOT / 'configs' / f'{stage}.json'),
                             sha256(ROOT / 'framereason/configs' / f'{stage}.json'))
        _, sft, _ = build_command('sft', '/model', '/data', '/run')
        _, grpo, _ = build_command('grpo', '/model', '/data', '/run', '/adapter700')
        self.assertEqual(sft['max_steps'], 750)
        self.assertEqual(sft['gradient_accumulation_steps'], 16)
        self.assertEqual(grpo['max_steps'], 3000)
        self.assertEqual(grpo['num_generations'], 4)
        self.assertEqual(grpo['max_completion_length'], 320)
        self.assertEqual(grpo['adapters'], grpo['ref_adapters'])
        self.assertEqual(grpo['beta'], 0)
        self.assertTrue(sft['freeze_vit'] and sft['freeze_aligner'])
        self.assertEqual(decode_values()['max_tokens'], 320)
        self.assertEqual(argument_values('/model')['max_pixels'], 50176)

    def test_grpo_cannot_start_without_sft(self):
        with self.assertRaises(ValueError):
            build_command('grpo', '/model', '/data', '/run')

    def test_dry_runs_need_no_model_no_cuda_no_swift(self):
        for args in (
            ['framereason.launch', 'sft', '--model', '/not-present', '--data', '/not-present', '--output', '/not-present'],
            ['framereason.infer', '--model', '/not-present', '--dataset', '/not-present', '--output', '/not-present'],
        ):
            result = subprocess.run([sys.executable, '-m', *args], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_guard_is_cpu_mocked_and_rejects_busy_card(self):
        fake = subprocess.CompletedProcess([], 0, stdout='0, GPU-TEST, 500\n')
        with patch('framereason.runtime.subprocess.run', return_value=fake):
            with self.assertRaises(RuntimeError):
                inspect_gpu(0, 'GPU-TEST')


if __name__ == '__main__':
    unittest.main()
