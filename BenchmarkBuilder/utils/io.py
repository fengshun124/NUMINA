import json
import os
from pathlib import Path

import click


def enum_files(
        file_path: str, file_ext: str = '.json',
        is_verbose: bool = False,
) -> list[str]:
    if os.path.isdir(file_path):
        files = [os.path.abspath(os.path.join(file_path, f))
                 for f in os.listdir(file_path) if f.endswith(file_ext)]
    elif os.path.isfile(file_path):
        files = [file_path]
    else:
        raise ValueError(f'Invalid file path: {os.path.abspath(file_path)}')

    if len(files) == 0:
        raise ValueError(f'No files with extension "{file_ext}" found in "{file_path}"')

    if is_verbose:
        print(f'Found {len(files)} files with extension "{file_ext}" in "{file_path}":')
        print('\n'.join(files) if len(files) < 10 else '\n'.join(files[:5] + ['...'] + files[-5:]))

    return files


def parse_json_text(
        json_text: str, required_fields: list[str],
) -> dict:
    """Parse the JSON text as a dictionary"""
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON format: {e}')

    data_dict = {}
    for field in required_fields:
        if field in json_data:
            data_dict[field] = json_data[field]
        else:
            raise ValueError(f'Missing field "{field}" in the JSON data')

    return data_dict


def load_json_file_as_dict(
        json_file_path: str, is_strict: bool = False
) -> dict:
    """Load the JSON file as a dictionary"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            if is_strict:
                raise ValueError(f'Invalid JSON format in {os.path.abspath(json_file_path)}: {e}')
            else:
                print(f'Invalid JSON format in {os.path.abspath(json_file_path)}: {e}')
                return {}


def export_dict_as_json_file(
        data_dict: dict, json_file_path: str,
) -> None:
    """Export the JSON file incrementally"""
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    # if JSON file does not exist or empty, export the data as JSON file
    if not os.path.isfile(json_file_path) or os.path.getsize(json_file_path) == 0:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump([data_dict], f, indent=4)
    # if JSON file exists and not empty, append the data to the JSON file
    else:
        try:
            with open(json_file_path, 'r+', encoding='utf-8') as f:
                f.seek(0)
                json_data = json.load(f)
                if not isinstance(json_data, list):
                    raise ValueError(f'Invalid JSON format in {json_file_path}')
                # append the data to the JSON file
                json_data.append(data_dict)
                f.seek(0)
                json.dump(json_data, f, indent=4)
                f.truncate()
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON format in {json_file_path}: {e}')


def confirm_overwrite_file(file_path: str | Path) -> bool:
    """Check if the file exists and ask for confirmation"""
    if os.path.isfile(file_path):
        if click.confirm(f'File "{file_path}" already exists. Overwrite?', default=False):
            os.remove(file_path)
            return True
        else:
            return False
    else:
        return True
