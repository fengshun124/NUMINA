import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.rule_based import RuleBasedQuestionGenerator


class CountQGenerator(RuleBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
    ):
        super().__init__(scene_stat_json_file)


def cli():
    pass


if __name__ == '__main__':
    cli()
