import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from BenchmarkBuilder.utils.chat_llm import chat_with_llm
from BenchmarkBuilder.utils.io import (
    confirm_overwrite_file,
    load_json_file_as_dict,
    export_dict_as_json_file
)


class LLMBasedQuestionGenerator(ABC):
    def __init__(
            self,
            question_json_file: str,
            rewrite_question_type: str,
            llm_model: str = 'qwen2.5:72b',
            llm_backend: Literal['ollama', 'openai'] = 'ollama',
            export_dir: str = './output/',
    ) -> None:
        """Initialize LLM-based question generator"""
        self.question_json_file = os.path.abspath(question_json_file)
        self.question_label = os.path.splitext(os.path.basename(self.question_json_file))[0]
        self.rewrite_question_type = rewrite_question_type

        # configure LLM model and backend
        self.llm_model = llm_model
        self.llm_backend = llm_backend
        print(f'Using "{llm_model}" model with "{llm_backend}" backend for rewriting the questions')

        # configure export path
        self.export_dir = Path(export_dir).absolute()
        print(f'Exporting the rewritten questions and failed rewritten questions to: {self.export_dir}')

        self.rewrite_json_file = os.path.join(
            self.export_dir, f'{self.rewrite_question_type}-{self.question_label}.json')
        self.fail_rewrite_json_file = os.path.join(
            self.export_dir, f'FAIL-{self.rewrite_question_type}-{self.question_label}.json')

        if not confirm_overwrite_file(self.rewrite_json_file):
            raise Exception(f'Aborted overwriting the existing file: {self.question_json_file}')
        if os.path.isfile(self.fail_rewrite_json_file):
            os.remove(self.fail_rewrite_json_file)

    def rewrite(
            self,
            max_retries: int = 5,
            question_key: str = 'prompt',
            answer_key: str = 'caption',
            meta_keys=None,
    ) -> None:
        """Rewrite the questions using the LLM model"""
        meta_keys = meta_keys or ['scene_id', 'obj_id']
        question_dicts = load_json_file_as_dict(self.question_json_file, is_strict=True)
        print(f'Loaded {len(question_dicts)} questions from: {self.question_json_file}')
        for idx, question_dict in tqdm(
                enumerate(question_dicts),
                desc='Rewriting', unit='Q', total=len(question_dicts)
        ):
            for attempt in range(max_retries):
                try:
                    rewrite_output_dict = self._rewrite_question(
                        question_dict[question_key],
                        question_dict[answer_key],
                        idx
                    )
                    if self._validate_rewritten_question(rewrite_output_dict):
                        export_dict_as_json_file({
                            **{key: question_dict[key] for key in meta_keys},
                            **rewrite_output_dict,
                        }, self.rewrite_json_file)
                    break
                except Exception as e:
                    print(f'[{attempt + 1}/{max_retries}] Failed to rewrite the question: {e}')
            else:
                print(f'Failed to rewrite the question after {max_retries} attempts')
                export_dict_as_json_file(question_dict, self.fail_rewrite_json_file)

    def _chat_with_llm(self, request_text: str) -> str:
        """Chat with the LLM model"""
        return chat_with_llm(request_text, self.llm_model, self.llm_backend)

    @abstractmethod
    def _rewrite_question(
            self,
            src_question: str,
            src_answer: str,
            question_set_idx: int,
            **kwargs
    ) -> dict[str, str | int | float]:
        """Generate a question using the LLM model"""
        pass

    @abstractmethod
    def _validate_rewritten_question(
            self,
            rewrite_output_dict: dict[str, str | int | float]
    ) -> bool:
        """Validate the response from the LLM model"""
        pass
