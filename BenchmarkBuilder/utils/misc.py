import json
import os
from contextlib import contextmanager
from multiprocessing.synchronize import Lock as ProcessLock
from threading import Lock as ThreadLock
from typing import Union, Optional

import click
from tqdm import tqdm


def parse_llm_response(response: str, fields: list[str]) -> dict:
    """Parse the LLM response as a JSON object"""
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid response: {response} as {e}')

    if not all(field in response_dict for field in fields):
        raise ValueError(f'Missing field(s) in response: {fields}')

    return response_dict


def enum_json_files(file_path: str, skip_confirm: bool) -> list[str]:
    """Form the list containing question JSON files based on the given path."""
    if os.path.isdir(file_path):
        question_jsons = [os.path.join(file_path, file)
                          for file in os.listdir(file_path) if file.endswith('.json')]
    elif os.path.isfile(file_path):
        question_jsons = [file_path]
    else:
        raise ValueError('Invalid input directory or file')

    print(f'The following JSON files will be processed\n' + '\n'.join(question_jsons))
    if not skip_confirm and not click.confirm('Proceed?', default=True):
        return []
    else:
        return question_jsons


@contextmanager
def acquire_lock(lock: Optional[Union[ThreadLock, ProcessLock]]):
    """Acquire the lock if available"""
    if lock:
        lock.acquire()
    try:
        yield
    finally:
        if lock:
            lock.release()


def export_json_file(
        data_dict: dict, target_json_file: str,
        lock: Optional[Union[ThreadLock, ProcessLock]] = None
) -> None:
    """Export the JSON file incrementally"""
    os.makedirs(os.path.dirname(target_json_file), exist_ok=True)
    with acquire_lock(lock):
        # if JSON file does not exist or empty, export the data as JSON file
        if not os.path.isfile(target_json_file) or os.path.getsize(target_json_file) == 0:
            with open(target_json_file, 'w', encoding='utf-8') as f:
                json.dump([data_dict], f, indent=4)
        # if JSON file exists and not empty, append the data to the JSON file
        else:
            try:
                with open(target_json_file, 'r+', encoding='utf-8') as f:
                    f.seek(0)
                    json_data = json.load(f)
                    if not isinstance(json_data, list):
                        raise ValueError(f'Invalid JSON format in {target_json_file}')
                    # append the data to the JSON file
                    json_data.append(data_dict)
                    f.seek(0)
                    json.dump(json_data, f, indent=4)
                    f.truncate()
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON format in {target_json_file}: {e}')


def load_json_file(file_path: str) -> dict:
    """Load the JSON file and return the data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data_dict = json.load(f)
            return data_dict
        except json.JSONDecodeError:
            print(f'Skipping {file_path}: Invalid JSON.')
            return {}


def validate_export_json_file(file_path: str) -> bool:
    """Check if the export file exists and confirm overwrite"""
    if os.path.isfile(file_path):
        if not click.confirm(f'File {file_path} exists. Overwrite?', default=False):
            print(f'Skipped {file_path} as export file exists.')
            return False
        else:
            print(f'{file_path} removed.')
            os.remove(file_path)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    return True


def process_question_jsons(
        question_json_path: str,
        rewrite_type: str, rewrite_method: callable,
        export_dir: str, export_prefix: str,
        skip_confirm: bool,
        max_retry: int,
        llm_model: str, llm_backend: str,
        **kwargs
):
    """Process the list of question JSON files with the given LLM-based rewrite methods"""
    question_jsons = enum_json_files(question_json_path, skip_confirm)
    if not question_jsons:
        print('No JSON files to process. Aborted.')
        return

    for question_json in question_jsons:
        question_dicts = load_json_file(question_json)
        if not question_dicts:
            print(f'Skipped {question_json} as no valid data found.')
            continue

        result_json_file = os.path.join(
            export_dir, f'{export_prefix}-{os.path.basename(question_json)}')
        if not validate_export_json_file(result_json_file):
            continue
        # JSON file for failed attempts
        failed_attempt_json_file = os.path.join(
            export_dir, f'failed-{export_prefix}-{os.path.basename(question_json)}')
        if os.path.exists(failed_attempt_json_file):
            os.remove(failed_attempt_json_file)

        fail_attempts = 0
        print(f'Generating {rewrite_type} for {os.path.basename(question_json)}...')
        with tqdm(total=len(question_dicts), desc=f'Processing ', ascii=' >=', unit='Q') as pbar:
            for idx, question_dict in enumerate(question_dicts):
                for attempt in range(max_retry):
                    try:
                        rewrite_question_dict = rewrite_method(
                            question_dict['prompt'],
                            question_dict['caption'],
                            llm_backend=llm_backend,
                            llm_model=llm_model,
                            **kwargs
                        )
                        # export the generated MCQ as JSON
                        export_json_file({
                            'scene_id': question_dict['scene_id'],
                            'obj_id': question_dict['obj_id'],
                            **rewrite_question_dict
                        }, result_json_file)
                        break
                    except ValueError as e:
                        print(f'Attempt {attempt + 1}/{max_retry} failed: {e}')
                else:
                    export_json_file(question_dict, failed_attempt_json_file)
                    fail_attempts += 1
                    print(f'Failed to generate {rewrite_type} for question {idx + 1} '
                          f'in {question_json} after {max_retry} attempts.')

                pbar.update(1)

        print(f'Generated {len(question_dicts) - fail_attempts} {rewrite_type} for {question_json}.')
        if fail_attempts > 0:
            print(f'Failed to generate {fail_attempts} {rewrite_type} for {question_json}. '
                  f'Refer to {failed_attempt_json_file} for details.')
