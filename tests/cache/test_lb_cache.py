# -*- coding: utf-8 -*-
import asyncio
import os
import tempfile

import diskcache
import pytest
import redis
from pytest_mock import MockerFixture

from lbgpt import ChatGPT
from lbgpt.types import ChatCompletionAddition


def setup_redis():
    redis_cache = redis.StrictRedis(
        host="localhost",
        port=6378,  # this is the port configured in the compose.yml file
    )

    redis_cache.flushdb()
    return redis_cache


def setup_diskcache():
    return diskcache.Cache(tempfile.TemporaryDirectory().name)


CACHES = [
    setup_diskcache,
    setup_redis
]


def _num_keys_in_cache(cache) -> int:
    if isinstance(cache, redis.StrictRedis):
        return cache.dbsize()
    elif hasattr(cache, "count"):
        return cache.count
    else:
        raise NotImplementedError


@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.vcr(ignore_localhost=True)
def test_chatgpt_cache(mocker: MockerFixture, cache):
    cache = cache()
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
        cache=cache,
    )

    # Setting the mocks
    cache_interaction = mocker.spy(lb, "_get_from_cache")
    request_interaction = mocker.spy(lb, "chat_completion")

    # run with an empty cache
    asyncio.run(lb.chat_completion_list([single_request_content], show_progress=False))

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(cache), "count": _num_keys_in_cache(cache)}

    # Getting from cache
    asyncio.run(lb.chat_completion_list([single_request_content], show_progress=False))

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    assert hash(cache) == cache_stats["hash"]
    assert _num_keys_in_cache(cache) == cache_stats["count"]

    # assert that chatgpt was requested only once
    assert request_interaction.call_count == 1


@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.vcr(ignore_localhost=True)
def test_chatgpt_cache_with_name_alias(mocker: MockerFixture, cache):
    cache = cache()

    messages = [
        {"role": "user", "content": "please respond with pong"},
    ]
    single_request_content_raw_name = dict(
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
        cache=cache,
    )

    # Setting the mocks
    cache_interaction = mocker.spy(lb, "_get_from_cache")
    request_interaction = mocker.spy(lb, "chat_completion")

    # run with an empty cache
    asyncio.run(lb.chat_completion_list([single_request_content_raw_name], show_progress=False))

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(cache), "count": _num_keys_in_cache(cache)}

    single_request_content_equivalent_name = dict(
        messages=messages,
        model="gpt-3.5-turbo",
        model_name_cache_alias="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    # Getting from cache
    asyncio.run(lb.chat_completion_list([single_request_content_equivalent_name], show_progress=False))

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    assert hash(cache) == cache_stats["hash"]
    assert _num_keys_in_cache(cache) == cache_stats["count"]

    # assert that chatgpt was requested only once
    assert request_interaction.call_count == 1
