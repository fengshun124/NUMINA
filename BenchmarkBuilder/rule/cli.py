import os
from abc import ABC
from typing import List, Callable

import click
from click.decorators import FC

from BenchmarkBuilder.utils.cli import BaseCLI
from BenchmarkBuilder.utils.io import enum_files, confirm_overwrite_file


class RuleBasedCLI(BaseCLI, ABC):
    def __init__(
            self,
            name: str,
            description: str,
    ) -> None:
        super().__init__(name, description)
        self.scene_stat_json_file = ''
        self.output_json_file = ''
        self.exclude_labels = ''

        method = self.execute
        method = click.command(name=name, help=description)(method)
        for decorator in reversed(self.common_options + self.specific_options):
            method = decorator(method)
        self.execute = method


    @property
    def common_options(self) -> List[Callable[[FC], FC]]:
        return super().common_options + [
            click.option(
                '--scene_stat_json_file',
                '-s',
                type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
                required=True,
                prompt='Path to the input scene stat JSON file',
                callback=lambda ctx, param, value: os.path.abspath(value),
                help='Path to the input scene stat JSON file'
            ),
            click.option(
                '--output_json_file',
                '-o',
                type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
                required=True,
                callback=lambda ctx, param, value: os.path.abspath(value),
                help='Path to the output JSON file'
            ),
            click.option(
                '--exclude_labels',
                '-e',
                type=str,
                multiple=True,
                callback=lambda ctx, param, value: list(value),
                help='Labels to exclude from the question generation',
                default=['wall', 'floor', 'ceiling', 'object'],
                show_default=True,
            ),
        ]

    def prepare_scene_jsons(self) -> List[str]:
        """Prepare list of input scene stat JSON files and confirm"""
        try:
            return enum_files(
                file_path=self.scene_stat_json_file,
                is_skip_confirm=self._is_skip_confirm
            )
        except (ValueError, click.Abort) as e:
            raise click.ClickException(str(e))

    def check_export_json(self) -> bool:
        """Check if export JSON file exists and handle overwrite"""
        if self._is_skip_confirm:
            if os.path.isfile(self.output_json_file):
                raise click.ClickException(
                    f'Target export JSON file {self.output_json_file} already exists. '
                )
            return True

        return confirm_overwrite_file(self.output_json_file)
