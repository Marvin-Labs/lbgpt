import json
from os import environ
from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

from lbgpt.lbgpt import LiteLlmRouter
from lbgpt.types import ChatCompletionAddition

messages = [
    {"role": "user", "content": "Roll one dice!"},
]
single_request_content = dict(
    messages=messages,
    temperature=0,
    max_tokens=200,
    top_p=1,
    request_timeout=10,
)

STANDARD_MODEL_TESTS = [
    ("openai/gpt-4o", "OPEN_AI_API_KEY", "gpt-4o-2024-08-06", None),
    ("gemini/gemini-2.0-flash", "GEMINI_API_KEY", "gemini-1.5-flash", None),
]


@pytest.mark.parametrize(
    "model_name, api_key, expected_model, params",
    STANDARD_MODEL_TESTS,
    ids=[x[0] for x in STANDARD_MODEL_TESTS],
)
# @pytest.mark.vcr(filter_query_parameters=["key"], match_on=("method", "scheme", "host", "port", "path", "query"), record_mode='new_episodes')
async def test_litellm_with_tools(model_name, api_key, expected_model, params):
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

    lb = LiteLlmRouter(model_list, tools=[{
        'type': 'function',
        'function':
        {'name': 'roll_dice', 'description': 'Roll `n_dice` 6-sided dice and return the results.', 'parameters': {'properties': {'n_dice': {'title': 'N Dice', 'type': 'integer'}}, 'required': ['n_dice'], 'type': 'object'}, 'annotations': None}}])
    res = await lb.chat_completion(model=model_name, **single_request_content)
    assert isinstance(res, ChatCompletionAddition)
    assert res.choices[0].message.tool_calls[0].function.name == "roll_dice"
    assert json.loads(res.choices[0].message.tool_calls[0].function.arguments).get("n_dice") == 1
    assert res.choices[0].finish_reason == "tool_calls"
