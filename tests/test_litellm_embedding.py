from os import environ

import pytest
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

from lbgpt.lbgpt import LiteLlmRouter
from lbgpt.types import EmbeddingResponseAddition

messages = [
    {"role": "user", "content": "please respond with pong"},
]
single_request_content = dict(
    input="the is a test for an embeddings string",
)

STANDARD_MODEL_TESTS = [
    (
        "openai/text-embedding-ada-002",
        "OPEN_AI_API_KEY",
        "text-embedding-ada-002",
        None,
    ),
    ("gemini/text-embedding-004", "GEMINI_API_KEY", "text-embedding-004", None),
]


@pytest.mark.parametrize(
    "model_name, api_key, expected_model, params",
    STANDARD_MODEL_TESTS,
    ids=[x[0] for x in STANDARD_MODEL_TESTS],
)
@pytest.mark.vcr(filter_query_parameters=["key"])
async def test_litellm_embedding(model_name, api_key, expected_model, params):
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
    res = await lb.embedding(model=model_name, **single_request_content)

    assert isinstance(res, EmbeddingResponseAddition)
    assert res.model == expected_model
    assert res.model_class == "LiteLlmRouter"
