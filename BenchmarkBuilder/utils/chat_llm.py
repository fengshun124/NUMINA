import os
import re
from typing import Literal


def _cleanup_response(response: str) -> str:
    """Remove non-Latin characters (e.g., Chinese characters) and spacing from response"""
    return re.sub(r'[\u4e00-\u9fff]+', '', response).replace('\n', '').replace('\r', '')


def _chat_ollama(request_text: str, llm_model: str = 'qwen2.5:72b') -> str:
    """Chat with LLM models hosted by ollama"""
    import ollama
    response = ollama.chat(
        model=llm_model,
        messages=[{'role': 'user', 'content': request_text}]
    )
    return _cleanup_response(response['message']['content'])


def _chat_openai(request_text: str, llm_model: str = 'gpt-4o-mini-2024-07-18') -> str:
    """Chat with LLM models compatible with OpenAI API"""
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()

    client = OpenAI(
        base_url=os.getenv('OPENAI_BASE_URL'),
        api_key=os.getenv('OPENAI_API_KEY')
    )
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": request_text}]
    )
    return _cleanup_response(response.choices[0].message.content)


def chat_with_llm(
        request_text: str,
        llm_model: str = 'qwen2.5:72b',
        llm_backend: str = 'ollama',
) -> str:
    """Helper function for LLM chatting"""
    match llm_backend:
        case 'ollama':
            return _chat_ollama(request_text, llm_model)
        case 'openai':
            return _chat_openai(request_text, llm_model)
        case _:
            raise ValueError('Invalid backend specified. Must be either "ollama" or "openai"')
