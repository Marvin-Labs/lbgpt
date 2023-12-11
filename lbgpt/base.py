# -*- coding: utf-8 -*-
import abc
import datetime
import sys
from logging import getLogger
import asyncio
import logging
from statistics import median
from typing import Any, Optional, Sequence
import openai
import openai.error
from tenacity import (
    after_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    AsyncRetrying,
)

from lbgpt.cache import make_hash_sha256
from lbgpt.usage import Usage, UsageStats

logger = getLogger(__name__)


class _BaseGPT(abc.ABC):
    def __init__(
        self,
        max_parallel_calls: int,
        cache: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
        max_usage_cache_size: Optional[int] = 1_000,
        limit_tpm: Optional[int] = None,
        limit_rpm: Optional[int] = None,
    ):
        # this is standard cache, i.e. it only checks for equal items
        self.cache = cache
        self.semantic_cache = semantic_cache

        self.max_parallel_calls = max_parallel_calls
        self.semaphore = self.refresh_semaphore()
        self.stop_after_attempts = stop_after_attempts
        self.stop_on_exception = stop_on_exception

        self.usage_cache_list = []
        self.max_usage_cache_size = max_usage_cache_size

        self.limit_tpm = limit_tpm
        self.limit_rpm = limit_rpm

    def add_usage_to_usage_cache(self, usage: Usage):
        # evict if the list is too long. Do this to protect memory usage if required
        if self.max_usage_cache_size and len(self.usage_cache_list) > self.max_usage_cache_size:
            self.usage_cache_list.pop(0)

        self.usage_cache_list.append(usage)

    def usage_cache_list_after_start_datetime(
        self, start_datetime: datetime.datetime
    ) -> list[Usage]:
        return [k for k in self.usage_cache_list if k.start_datetime > start_datetime]

    def get_usage_stats(self, include_usage_reservation: bool = False) -> UsageStats:
        current_usage_tokens = 0
        current_usage_requests = 0

        if include_usage_reservation:
            # Estimating current usage as the expected number of tokens times the number of
            # currently outstanding requests (which are tracked in the semaphore)
            current_usage_tokens = (
                self.semaphore._value * self.expected_tokens_per_request()
            )
            current_usage_requests = self.semaphore._value

        cache_list_after_start_datetime = self.usage_cache_list_after_start_datetime(
            datetime.datetime.now() - datetime.timedelta(seconds=60)
        )

        return UsageStats(
            tokens=sum(
                [
                    k.input_tokens + k.output_tokens
                    for k in cache_list_after_start_datetime
                ]
            )
            + current_usage_tokens,
            requests=len(cache_list_after_start_datetime) + current_usage_requests,
        )

    def expected_tokens_per_request(self) -> int:
        """Returns the expected number of tokens per usage
        We are returning the following (in subsequent order of preference):
        (1) the median number of tokens per usage in the cache
        (2) 10% of the limit_tpm
        (3) zero otherwise
        """
        if len(self.usage_cache_list) > 0:
            return int(
                median([k.input_tokens + k.output_tokens for k in self.usage_cache_list])
            )

        if self.limit_tpm:
            return int(self.limit_tpm * 0.1)
        else:
            return 0

    def headroom(self) -> int:
        """Returns the number of tokens remaining in the current minute"""
        cur_usage = self.get_usage_stats()

        if self.limit_tpm:
            headroom_tpm = (
                self.limit_tpm - cur_usage.tokens - self.expected_tokens_per_request()
            )
        else:
            headroom_tpm = sys.maxsize

        if self.limit_rpm:
            headroom_rpm = (
                self.expected_tokens_per_request()
                * (self.limit_rpm - cur_usage.requests - 1)
            )
        else:
            headroom_rpm = sys.maxsize

        return min([headroom_tpm, headroom_rpm])

    def refresh_semaphore(self) -> asyncio.Semaphore:
        semaphore = asyncio.Semaphore(value=self.max_parallel_calls)
        self.semaphore = semaphore
        return semaphore

    @abc.abstractmethod
    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        raise NotImplementedError

    async def chat_completion_list(
        self,
        chatgpt_chat_completion_request_body_list: list[dict],
    ) -> Sequence[openai.ChatCompletion]:
        self.refresh_semaphore()

        return await asyncio.gather(
            *[
                self.cached_chat_completion(**content)
                for content in chatgpt_chat_completion_request_body_list
            ]
        )



    async def cached_chat_completion(self, **kwargs) -> Optional[openai.ChatCompletion]:
        # this is standard cache. We are always trying standard cache first
        if self.cache is not None:
            hashed = make_hash_sha256(kwargs)
            if hashed in self.cache:
                logger.debug("standard cache hit")
                return self.cache[hashed]

        # if the item is not in the standard cache, we are trying the semantic cache (if available)
        # we are currently only supporting semantic cache for FAISS models
        # TODO: ADD


        try:
            async for attempt in AsyncRetrying(
                retry=(
                    retry_if_exception_type(openai.error.Timeout)
                    | retry_if_exception_type(openai.error.RateLimitError)
                    | retry_if_exception_type(openai.error.TryAgain)
                    | retry_if_exception_type(openai.error.APIConnectionError)
                    | retry_if_exception_type(openai.error.APIError)
                ),
                wait=wait_random_exponential(min=5, max=60),
                stop=stop_after_attempt(self.stop_after_attempts)
                if self.stop_after_attempts is not None
                else None,
                after=after_log(logger, logging.WARNING),
            ):
                with attempt:
                    out = await self.chat_completion(**kwargs)

        except Exception as e:
            if self.stop_on_exception:
                raise e
            else:
                logger.exception(e)
                return None

        if self.cache is not None:
            self.cache[hashed] = out

        return out
