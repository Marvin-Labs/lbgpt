import asyncio
import os
import random

from lbgpt import LoadBalancedGPT
from pytest_mock import MockerFixture


random.seed(42)


def test_chatgpt_async(mocker: MockerFixture):
    messages = [
        {"role": "user", "content": "please respond with pong"},
    ]
    single_request_content = dict(
        messages=messages,
        model="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    lb = LoadBalancedGPT(
        openai_api_key=os.environ["OPEN_AI_API_KEY"],
        azure_api_key=os.environ["OPEN_AI_AZURE_API_KEY"],
        azure_api_base=os.environ["OPEN_AI_AZURE_URI"],
        azure_model_map={
            "gpt-3.5-turbo-0613": os.environ["OPENAI_AZURE_DEPLOYMENT_ID"]
        },
    )

    azure = mocker.spy(lb.azure, "chat_completion")
    openai = mocker.spy(lb.openai, "chat_completion")

    res = asyncio.run(lb.chat_completion_list([single_request_content] * 5))

    assert len(res) == 5
    for k in res:
        assert "pong" in k.choices[0].message.content.lower()

    assert azure.call_count >= 1
    assert openai.call_count >= 1
    assert azure.call_count + openai.call_count == 5