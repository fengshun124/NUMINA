# Benchmark Builder

- OpenAI API

  To use LLM with the OpenAI API, set up a .env file in [utils](utils) with the OpenAI API key and base URL with the
  following format:

  ```bash
  OPENAI_BASE_URL=<your_openai_base_url>
  OPENAI_API_KEY=<your_openai_api_key>
  ```

  Replace <openai_base_url> and <openai_api_key> with the actual OpenAI-compatible API base URL and API key.

## Prompt and Caption

**Example**

Consider the following question and answer pair:

> Q: What is the capital of France?
> A: Paris

- For normal QA, the prompt and caption for fine-tuning would be like

  ```json
  {
      "prompt" : "What is the capital of France? Answer the question with a single word or phrase.",
      "caption" : "Paris"
  }
  ```

  or

  ```json
  {
      "prompt" : "What is the capital of France? A) Paris B) London C) Berlin D) Rome. Please select the correct option.",
      "caption" : "A"
  }
  ```

- For reasoning QA, the final answer would be wrapped in `<<answer:>>` tag.
  In this case, the prompt and caption for fine-tuning would be like

  ```json
  {
      "prompt" : "What is the capital of France? Answer the question with a chain of thought (CoT) reasoning, and wrap the final answer in <<answer:>> tag.",
      "CoT_caption" : "Paris is the capital of France. <<answer:Paris>>"
  }
  ```

  To catch the final answer, the regex pattern `<<answer:(.*?)>>` can be used. 