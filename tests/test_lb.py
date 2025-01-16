# -*- coding: utf-8 -*-
import asyncio
import os
import random

import pytest
from pytest_mock import MockerFixture

from lbgpt import AzureGPT, ChatGPT, LoadBalancedGPT, MultiLoadBalancedGPT
from lbgpt.types import ChatCompletionAddition


@pytest.mark.vcr
async def test_lb_async_random(mocker: MockerFixture):
    random.seed(42)

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
        auto_cache=False,
    )

    azure = mocker.spy(lb.azure, "chat_completion")
    openai = mocker.spy(lb.openai, "chat_completion")

    res = await lb.achat_completion_list(
        [single_request_content] * 5, show_progress=False
    )

    assert len(res) == 5
    for k in res:
        assert "pong" in k.choices[0].message.content.lower()
        assert isinstance(k, ChatCompletionAddition)

    assert azure.call_count >= 1
    assert openai.call_count >= 1
    assert azure.call_count + openai.call_count == 5


@pytest.mark.vcr
async def test_lbgpt_max_headroom():
    random.seed(42)

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

    model_openai = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        limit_tpm=1000,
        auto_cache=False,
    )

    model_azure = AzureGPT(
        stop_after_attempts=1,
        stop_on_exception=True,
        api_key=os.environ["OPEN_AI_AZURE_API_KEY"],
        azure_api_base=os.environ["OPEN_AI_AZURE_URI"],
        azure_model_map={
            "gpt-3.5-turbo-0613": os.environ["OPENAI_AZURE_DEPLOYMENT_ID"]
        },
        limit_tpm=100,
        auto_cache=False,
    )

    lb = MultiLoadBalancedGPT(
        gpts=[model_openai, model_azure],
        allocation_function="max_headroom",
        auto_cache=False,
    )

    res = await lb.achat_completion_list(
        [single_request_content] * 5, show_progress=False
    )

    assert len(await lb.usage_cache_list) == 5
    assert len(await lb.gpts[0].usage_cache_list) == 5
    assert len(await lb.gpts[1].usage_cache_list) == 0

    assert len(res) == 5
    for k in res:
        assert "pong" in k.choices[0].message.content.lower()
        assert isinstance(k, ChatCompletionAddition)


@pytest.mark.vcr
async def test_chatgpt_async(mocker: MockerFixture):
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

    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        auto_cache=False,
    )

    openai = mocker.spy(lb, "chat_completion")

    res = await lb.achat_completion_list(
        [single_request_content] * 5, show_progress=False
    )

    assert len(await lb.usage_cache_list) == 5

    assert len(res) == 5
    for k in res:
        assert "pong" in k.choices[0].message.content.lower()
        assert isinstance(k, ChatCompletionAddition)

    assert openai.call_count >= 5


@pytest.mark.vcr
async def test_azure_async(mocker: MockerFixture):
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

    lb = AzureGPT(
        stop_after_attempts=1,
        stop_on_exception=True,
        api_key=os.environ["OPEN_AI_AZURE_API_KEY"],
        azure_api_base=os.environ["OPEN_AI_AZURE_URI"],
        azure_model_map={
            "gpt-3.5-turbo-0613": os.environ["OPENAI_AZURE_DEPLOYMENT_ID"]
        },
        auto_cache=False,
    )

    azure = mocker.spy(lb, "chat_completion")

    res = await lb.achat_completion_list(
        [single_request_content] * 5, show_progress=False
    )

    assert len(await lb.usage_cache_list) == 5

    assert len(res) == 5
    for k in res:
        assert "pong" in k.choices[0].message.content.lower()
        assert isinstance(k, ChatCompletionAddition)

    assert azure.call_count >= 5


@pytest.mark.vcr
def test_chatgpt_async_multiple_starts(mocker: MockerFixture):
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

    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
    )

    openai = mocker.spy(lb, "chat_completion")

    res = asyncio.run(
        lb.achat_completion_list([single_request_content] * 2, show_progress=False)
    )

    res2 = asyncio.run(
        lb.achat_completion_list([single_request_content] * 2, show_progress=False)
    )

    assert len(res) + len(res2) == 4
