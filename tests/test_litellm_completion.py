from os import environ
from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

from lbgpt.lbgpt import LiteLlmRouter
from lbgpt.types import ChatCompletionAddition

messages = [
    {"role": "user", "content": "please respond with pong"},
]
single_request_content = dict(
    messages=messages,
    temperature=0,
    max_tokens=50,
    top_p=1,
    request_timeout=10,
)

STANDARD_MODEL_TESTS = [
    ("openai/gpt-4o", "OPEN_AI_API_KEY", "gpt-4o-2024-08-06", None),
    ("gemini/gemini-1.5-flash", "GEMINI_API_KEY", "gemini-1.5-flash", None),
    (
        "vertex_ai/gemini-1.5-flash",
        None,
        "gemini-1.5-flash",
        {"vertex_credentials": Path("./vertex_dev_key.json").read_text()},
    ),
    ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat", None),
    (
        "openai/meta-llama/Llama-3.2-1B-Instruct",
        "NEBIUS_API_KEY",
        "meta-llama/Llama-3.2-1B-Instruct",
        {
            "api_base": "https://api.studio.nebius.ai/v1",
            "custom_llm_provider ": "nebius",
        },
    ),
    (
        "openai/Qwen/Qwen2.5-72B-Instruct",
        "NEBIUS_API_KEY",
        "Qwen/Qwen2.5-72B-Instruct",
        {
            "api_base": "https://api.studio.nebius.ai/v1",
            "custom_llm_provider ": "nebius",
        },
    ),
    (
        "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "TOGETHER_AI_API_KEY",
        "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        None,
    ),
    (
        "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        "TOGETHER_AI_API_KEY",
        "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        None,
    ),
]


@pytest.mark.parametrize(
    "model_name, api_key, expected_model, params",
    STANDARD_MODEL_TESTS,
    ids=[x[0] for x in STANDARD_MODEL_TESTS],
)
@pytest.mark.vcr(filter_query_parameters=["key"], match_on=("method", "scheme", "host", "port", "path", "query"))
async def test_litellm_standard_models(model_name, api_key, expected_model, params):
    if params is None:
        params = {}
    litellm_params: LiteLLMParamsTypedDict = {
        "model": model_name,
        "max_retries": 0,
        **params,
    }
    if api_key is not None:
        litellm_params["api_key"] = environ.get(api_key, api_key)

    model_list: list[DeploymentTypedDict] = [
        {
            "model_name": model_name,
            "litellm_params": litellm_params,
        }
    ]

    lb = LiteLlmRouter(model_list)
    res = await lb.chat_completion(model=model_name, **single_request_content)
    assert isinstance(res, ChatCompletionAddition)
    assert res.model == expected_model
    assert res.model_class == "LiteLlmRouter"
