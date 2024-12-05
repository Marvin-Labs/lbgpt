from os import environ
from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import DeploymentTypedDict

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


@pytest.mark.vcr
async def test_litellm_chatgpt():
    model_list: list[DeploymentTypedDict] = [
        {
            'model_name': '*',
            'litellm_params': {
                'model': 'openai/*',
                'api_key': environ["OPEN_AI_API_KEY"]
            }
        }
    ]

    lb = LiteLlmRouter(model_list)
    res = await lb.chat_completion(model='gpt-4o', **single_request_content)
    assert isinstance(res, ChatCompletionAddition)
    assert res.model == 'gpt-4o-2024-08-06'
    assert res.model_class == 'LiteLlmRouter'


@pytest.mark.vcr
async def test_litellm_gemini():
    model_list: list[DeploymentTypedDict] = [
        {
            'model_name': 'gemini/*',
            'litellm_params': {
                'model': 'gemini/*',
                'api_key': environ["GEMINI_API_KEY"]
            }
        }
    ]

    lb = LiteLlmRouter(model_list)
    res = await lb.chat_completion(model='gemini/gemini-1.5-flash', **single_request_content)
    assert isinstance(res, ChatCompletionAddition)
    assert res.model == 'gemini-1.5-flash'
    assert res.model_class == 'LiteLlmRouter'


@pytest.mark.vcr(match_on=("method", "scheme", "host", "port", "path", "query"))
async def test_litellm_vertex():
    model_list: list[dict[str, Any]] = [
        {
            'model_name': 'gemini/*',
            'litellm_params': {
                'model': 'vertex_ai/*',
                'vertex_credentials': Path('./vertex_dev_key.json').read_text()
            }
        }
    ]

    lb = LiteLlmRouter(model_list)
    res = await lb.chat_completion(model='gemini/gemini-1.5-flash', **single_request_content)
    assert isinstance(res, ChatCompletionAddition)
    assert res.model == 'gemini-1.5-flash'
    assert res.model_class == 'LiteLlmRouter'

