from os import environ
from pathlib import Path
from typing import Any

import pytest
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

from lbgpt.lbgpt import LiteLlmRouter
from lbgpt.types import ChatCompletionAddition


@pytest.mark.vcr
def test_gemini_via_litellm_sync():
    single_request_content = dict(
        temperature=0,
        max_tokens=50,
        top_p=1,
        request_timeout=10,
    )

    def prompt_to_request(list_of_prompt, model_name):
        return [
            {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                **single_request_content
            }
            for prompt in list_of_prompt
        ]

    model_name = 'gemini/gemini-2.0-flash'
    model_list: list[DeploymentTypedDict] = [
        {
            "model_name": model_name,
            "litellm_params": {
        "model": model_name,
        "max_retries": 0,
        'api_key': environ.get('GEMINI_API_KEY', None),
    },
        }
    ]

    lb = LiteLlmRouter(model_list)
    prompts = [
        'ping',
        'ping ping',
    ]

    res = lb.chat_completion_list(prompt_to_request(prompts, model_name))
    for r in res:
        assert isinstance(r, ChatCompletionAddition)
