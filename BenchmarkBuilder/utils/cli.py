from abc import ABC, abstractmethod
from typing import Callable

import click
from click.decorators import FC

NUMINA_BANNER = r"""

███╗   ██╗ ██╗   ██╗ ███╗   ███╗ ██╗ ███╗   ██╗  █████╗ 
████╗  ██║ ██║   ██║ ████╗ ████║ ██║ ████╗  ██║ ██╔══██╗
██╔██╗ ██║ ██║   ██║ ██╔████╔██║ ██║ ██╔██╗ ██║ ███████║
██║╚██╗██║ ██║   ██║ ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██╔══██║
██║ ╚████║ ╚██████╔╝ ██║ ╚═╝ ██║ ██║ ██║ ╚████║ ██║  ██║
╚═╝  ╚═══╝  ╚═════╝  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝  ╚═╝

=============================================================
Natural Understanding of Multi-dimensional 
Intelligence and Numerical Abilities

Benchmark Builder CLI
=============================================================
"""


class BaseCLI(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._is_display_banner = True
        self._is_skip_confirm = False

    @staticmethod
    def display_arguments(**kwargs):
        """Display all arguments being used"""
        click.echo("\nUsing arguments:")
        for key, value in kwargs.items():
            click.echo(f"  {key}: {value}")
        click.echo("")

    @property
    def skip_confirm(self) -> bool:
        return self._is_skip_confirm

    @skip_confirm.setter
    def skip_confirm(self, value: bool) -> None:
        self._is_skip_confirm = value

    @property
    def common_options(self) -> list[Callable[[FC], FC]]:
        return [
            click.option(
                '--skip_confirm',
                is_flag=True, default=False,
                help='Skip confirmation prompt before execution',
                callback=lambda ctx, param, value: setattr(ctx, 'skip_confirm', value)
            )
        ]

    @property
    @abstractmethod
    def specific_options(self) -> list[Callable[[FC], FC]]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs) -> None:
        raise NotImplementedError

    def confirm_execution(self, **kwargs) -> bool:
        """Present received arguments and confirm execution (if not skipped)"""
        click.echo(f' {self.name} '.center(60, '-'))
        click.echo(self.description)
        click.echo()

        # Use existing display_arguments method
        self.display_arguments(**kwargs)

        # confirm execution
        if self.skip_confirm:
            return True
        return click.confirm('Proceed with execution?', default=True)
