# -*- coding: utf-8 -*-
import abc
from logging import getLogger
import asyncio
import logging
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

logger = getLogger(__name__)


class _BaseGPT(abc.ABC):
    def __init__(
        self,
        max_parallel_calls: int,
        cache: Optional[Any] = None,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        self.cache = cache
        self.max_parallel_calls = max_parallel_calls
        self.semaphore = self.refresh_semaphore()
        self.stop_after_attempts = stop_after_attempts
        self.stop_on_exception = stop_on_exception

    def refresh_semaphore(self):
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
        if self.cache is not None:
            hashed = make_hash_sha256(kwargs)
            if hashed in self.cache:
                logger.debug("cache hit")
                return self.cache[hashed]

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
