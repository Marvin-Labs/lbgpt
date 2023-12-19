# -*- coding: utf-8 -*-
import asyncio
import os
import tempfile

import diskcache
import pytest
from pytest_mock import MockerFixture

from lbgpt import ChatGPT
from lbgpt.types import ChatCompletionAddition

CACHES = [diskcache.Cache(tempfile.TemporaryDirectory().name)]


@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.vcr
def test_chatgpt_cache(mocker: MockerFixture, cache):
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

    # Setting the cache
    cache_interaction = mocker.spy(lb, "_request_from_cache")

    # run with an empty cache
    asyncio.run(lb.chat_completion_list([single_request_content], show_progress=False))

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(cache)}

    if hasattr(cache, "count"):
        cache_stats["count"] = cache.count

    # Getting from cache
    asyncio.run(lb.chat_completion_list([single_request_content], show_progress=False))

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    if hasattr(cache, "count"):
        assert cache.count == cache_stats["count"]
    assert hash(cache) == cache_stats["hash"]
