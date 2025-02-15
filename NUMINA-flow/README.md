# NUMINA-flow

## Introduction

As part of `NUMINA`, `NUMINA-flow` is a specialized pipeline for generating numerical reasoning questions from 3D scene data. Built on the `ScanNet` and `ScanQA` dataset, it employs a dual-track generation approach:

1. **LLM-based Track**: Rewrites Short-Answer Questions from `ScanQA` into structured formats
2. **Rule-based Track**: Generates questions directly from annotated `ScanNet` scenes

This complementary approach ensures both linguistic quality and factual accuracy in the generated question-answer pairs.

## Requirements

- Python 3.10 or later
- One of the following LLM backends:
  - OpenAI-compatible API access
  - Local LLM deployment via Ollama

## Configuration

### Environment Setup

Choose one of the following installation methods:

- **virtualenv + pip**

  ```bash
  # create and activate a virtual environment
  python -m venv venv
  source venv/bin/activate
  # install dependencies from requirements.txt
  pip install -r requirements.txt
  ```

- **conda/mamba**

  **For `mamba` users, replace `conda` with `mamba` in the following commands*:

  ```bash
  # Create and activate environment
  conda create -n numina-flow python=3.10
  conda activate numina-flow
  pip install -r requirements.txt
  ```

### LLM Backend Setup

`NUMINA-flow` supports two LLM backends: OpenAI API and Ollama. 

- **LLMs with OpenAI API**

  To use the OpenAI API, set up a .env file in [utils](./flow/utils) with the OpenAI API key and base URL with the following format:

  ```bash
  OPENAI_BASE_URL=<your_openai_base_url>
  OPENAI_API_KEY=<your_openai_api_key>
  ```

  Replace <openai_base_url> and <openai_api_key> with the actual OpenAI-compatible API base URL and API key.

- **LLMs with Ollama**

  To use Ollama, install the Ollama CLI and pull the required LLMs:

  ```bash
   # install ollama
   curl https://ollama.ai/install.sh | sh
   # pull default models
   ollama pull qwen2.5:72b
   ```

   For more available LLMs, visit the [Ollama Model Hub](https://ollama.ai/models).