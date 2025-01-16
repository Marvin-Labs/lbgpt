# -*- coding: utf-8 -*-
import asyncio
import os
import tempfile
from typing import Optional

import boto3
import diskcache
import pytest
import redis
from pytest_mock import MockerFixture

from lbgpt import ChatGPT
from lbgpt.caches.s3 import S3Cache
from lbgpt.types import ChatCompletionAddition
from cachetools import Cache


def setup_redis():
    redis_cache = redis.StrictRedis(
        host="localhost",
        port=6378,  # this is the port configured in the compose.yml file
    )

    redis_cache.flushdb()
    return redis_cache


def setup_diskcache():
    return diskcache.Cache(tempfile.TemporaryDirectory().name)


def setup_cachetools():
    return Cache(maxsize=1_000)


def setup_s3_cache():
    bucket = "data.lbgpt.com"
    prefix = "lbgpt-test/"

    s3_client = boto3.client(
        "s3",
        endpoint_url=' http://localhost:19900',
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )

    # we want to delete all keys before proceeding
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        delete_batch_size = 1000  # Maximum limit for delete_objects

        # Paginate through the S3 bucket
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:  # Check if there are objects to delete
                keys_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]

                # Delete in batches of 1000
                for i in range(0, len(keys_to_delete), delete_batch_size):
                    response = s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": keys_to_delete[i:i + delete_batch_size]},
                    )
                    deleted = response.get("Deleted", [])
                    print(f"Deleted {len(deleted)} objects from bucket '{bucket}'.")

    except Exception as e:
        print(f"Error when deleting objects: {str(e)}")

    return S3Cache(s3_client=s3_client, bucket=bucket, prefix=prefix)


CACHES = [
    setup_cachetools,
    setup_diskcache,
    setup_redis,
    setup_s3_cache,
]


def _num_keys_in_cache(cache) -> Optional[int]:
    if isinstance(cache, redis.StrictRedis):
        return cache.dbsize()
    elif hasattr(cache, "count"):
        return cache.count
    else:
        return None


def _hash_in_cache(cache) -> Optional[int]:
    try:
        return hash(cache)
    except TypeError:
        return cache.currsize


@pytest.mark.parametrize("cache", CACHES)
@pytest.mark.vcr(ignore_localhost=True)
def test_chatgpt_cache(mocker: MockerFixture, cache):
    cache = cache()

    messages = [
        {"role": "user", "content": "please respond with pong"},
    ]
    single_request_content = dict(
        messages=messages,
        model="gpt-4o",
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
    cache_interaction = mocker.spy(lb, "_get_from_standard_cache")
    request_interaction = mocker.spy(lb, "chat_completion")

    # run with an empty cache
    asyncio.run(lb.cached_chat_completion(single_request_content))

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": _hash_in_cache(cache), "count": _num_keys_in_cache(cache)}

    # Getting from cache
    asyncio.run(lb.cached_chat_completion(single_request_content))

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    assert _hash_in_cache(cache) == cache_stats["hash"]
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
        model="gpt-4o",
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
    cache_interaction = mocker.spy(lb, "_get_from_standard_cache")
    request_interaction = mocker.spy(lb, "chat_completion")

    # run with an empty cache
    asyncio.run(lb.cached_chat_completion(single_request_content_raw_name))

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": _hash_in_cache(cache), "count": _num_keys_in_cache(cache)}

    single_request_content_equivalent_name = dict(
        messages=messages,
        model="gpt-4o-mini",
        model_name_cache_alias="gpt-4o",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )

    # Getting from cache
    asyncio.run(lb.cached_chat_completion(single_request_content_equivalent_name))

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    assert _hash_in_cache(cache) == cache_stats["hash"]

    if cache_stats['count']:
        assert _num_keys_in_cache(cache) == cache_stats["count"]

    # assert that chatgpt was requested only once
    assert request_interaction.call_count == 1
