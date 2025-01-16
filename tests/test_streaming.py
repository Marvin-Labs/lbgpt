from os import environ
from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import DeploymentTypedDict

from lbgpt.lbgpt import LiteLlmRouter
from lbgpt.types import ChatCompletionAddition

messages = [
    {
        "role": "user",
        "content": "please return the first 50 words of the US declaration of independence",
    },
]
single_request_content = dict(
    messages=messages,
    temperature=0,
    max_tokens=500,
    top_p=1,
    request_timeout=10,
)


async def test_streaming():
    model_list: list[DeploymentTypedDict] = [
        {
            "model_name": "*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": environ["OPEN_AI_API_KEY"],
            },
        }
    ]

    lb = LiteLlmRouter(model_list)
    res = await lb.chat_completion(
        model="gpt-4o", stream=True, **single_request_content
    )
    assert isinstance(res, ChatCompletionAddition)
    assert res.model == "gpt-4o"
    assert res.model_class == "LiteLlmRouter"
