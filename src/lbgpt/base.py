# -*- coding: utf-8 -*-
import abc
import asyncio
import datetime
import json
import logging
import sys
from logging import getLogger
from statistics import median
from typing import Any, Callable, Optional, Sequence

import nest_asyncio
import openai
from openai._compat import model_parse
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lbgpt.cache import make_hash_chatgpt_request
from lbgpt.types import ChatCompletionAddition, EmbeddingResponseAddition, ResponseTypes, EmbeddingAddition
from lbgpt.usage import Usage, UsageStats
from lbgpt.utils import convert_to_dictionary

logger = getLogger(__name__)


def after_logging(
    logger_level: int = logging.WARNING,
    logger_exception: bool = True,
) -> Callable[[RetryCallState], None]:
    def logging_function(retry_state: RetryCallState) -> None:
        with logging_redirect_tqdm():
            exception = retry_state.outcome.exception()
            logger.log(
                level=logger_level,
                msg=f"Retrying request after {'%0.2f' % retry_state.seconds_since_start}s as attempt {retry_state.attempt_number} ended with `{repr(exception)}`",
            )

            if logger_exception:
                logger.exception(exception)
                if hasattr(exception, "request"):
                    logger.error(exception.request)

    return logging_function


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
        propagate_semantic_cache_to_standard_cache: bool = False,
        auto_cache: bool = True,
    ):
        # this is standard cache, i.e. it only checks for equal items
        self.cache = cache

        if self.cache is None and auto_cache:
            from cachetools import LRUCache

            self.cache = LRUCache(maxsize=max_usage_cache_size * 100)

        self.semantic_cache = semantic_cache

        self.max_parallel_calls = max_parallel_calls
        self.stop_after_attempts = stop_after_attempts
        self.stop_on_exception = stop_on_exception

        self._usage_cache_list = []
        self.max_usage_cache_size = max_usage_cache_size

        self.limit_tpm = limit_tpm
        self.limit_rpm = limit_rpm

        self.propagate_semantic_cache_to_standard_cache = (
            propagate_semantic_cache_to_standard_cache
        )

        self.semaphore_chatgpt = asyncio.Semaphore(self.max_parallel_calls)
        self.semaphore_standard_cache = asyncio.Semaphore(self.max_parallel_calls)
        self.semaphore_semantic_cache = asyncio.Semaphore(self.max_parallel_calls)

    def request_setup(self):
        pass

    @property
    async def usage_cache_list(self) -> list[Usage]:
        return self._usage_cache_list

    async def add_usage_to_usage_cache(self, usage: Usage):
        # evict if the list is too long. Do this to protect memory usage if required
        if (
            self.max_usage_cache_size
            and len(self._usage_cache_list) > self.max_usage_cache_size
        ):
            self._usage_cache_list.pop(0)

        self._usage_cache_list.append(usage)

    async def usage_cache_list_after_start_datetime(
        self, start_datetime: datetime.datetime
    ) -> list[Usage]:
        return [k for k in self._usage_cache_list if k.start_datetime > start_datetime]

    async def get_usage_stats(
        self, include_usage_reservation: bool = False
    ) -> UsageStats:
        current_usage_tokens = 0
        current_usage_requests = 0

        if include_usage_reservation:
            # Estimating current usage as the expected number of tokens times the number of
            # currently outstanding requests (which are tracked in the semaphore)
            current_usage_tokens = (
                self.semaphore_chatgpt._value * await self.expected_tokens_per_request()
            )
            current_usage_requests = self.semaphore_chatgpt._value

        cache_list_after_start_datetime = (
            await self.usage_cache_list_after_start_datetime(
                datetime.datetime.now() - datetime.timedelta(seconds=60)
            )
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

    async def expected_tokens_per_request(self) -> int:
        """Returns the expected number of tokens per usage
        We are returning the following (in subsequent order of preference):
        (1) the median number of tokens per usage in the cache
        (2) 10% of the limit_tpm
        (3) zero otherwise
        """
        if len(self._usage_cache_list) > 0:
            return int(
                median(
                    [k.input_tokens + k.output_tokens for k in self._usage_cache_list]
                )
            )

        if self.limit_tpm:
            return int(self.limit_tpm * 0.1)
        else:
            return 0

    async def headroom(self) -> int:
        """Returns the number of tokens remaining in the current minute"""
        cur_usage = await self.get_usage_stats()

        if self.limit_tpm:
            headroom_tpm = (
                self.limit_tpm
                - cur_usage.tokens
                - await self.expected_tokens_per_request()
            )
        else:
            headroom_tpm = sys.maxsize

        if self.limit_rpm:
            headroom_rpm = await self.expected_tokens_per_request() * (
                self.limit_rpm - cur_usage.requests - 1
            )
        else:
            headroom_rpm = sys.maxsize

        return min([headroom_tpm, headroom_rpm])

    @abc.abstractmethod
    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        raise NotImplementedError

    async def achat_completion_list(
        self,
        chatgpt_chat_completion_request_body_list: list[dict],
        show_progress=True,
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
    ) -> Sequence[ChatCompletionAddition]:
        # we are setting up the semaphore here, so that we can refresh it if required
        # we are rate limiting the three things we care about: gpt requests, standard cache requests,
        # and semantic cache requests.

        tasks = [
            self._cached_chat_completion_with_index(
                logging_level=logging_level,
                logging_exception=logging_exception,
                content=content,
                index=index,
            )
            for index, content in enumerate(chatgpt_chat_completion_request_body_list)
        ]

        results: list[(ChatCompletionAddition | Exception | None)] = [None] * len(
            tasks
        )  # To store results in the correct order

        # Use tqdm to track progress
        for task in tqdm.as_completed(
            tasks, total=len(tasks), disable=not show_progress
        ):
            result_with_index = await task
            index, result = result_with_index
            results[index] = result

        return results

    def chat_completion_list(self, chatgpt_chat_completion_request_body_list, **kwargs):
        nest_asyncio.apply()
        return asyncio.run(
            self.achat_completion_list(
                chatgpt_chat_completion_request_body_list=chatgpt_chat_completion_request_body_list,
                **kwargs,
            )
        )

    async def _get_from_standard_cache(self, hashed: str) -> Optional[ResponseTypes | dict]:
        if self.cache is not None:
            logger.debug(f"trying to get {hashed} from standard cache")

            if hasattr(self.cache, "aget"):
                out = await self.cache.aget(hashed)
            elif hasattr(self.cache, "get"):
                out = self.cache.get(hashed)
            else:
                out = self.cache[hashed]

            if out is not None:
                logger.debug(f"found request {hashed} in standard cache")
                out_dict = json.loads(out)
                out_dict["is_cached"] = True

                out_type = out_dict.get("object")

                if out_type == "chat.completion":
                    return model_parse(ChatCompletionAddition, out_dict)
                elif out_type == "embedding":
                    return out_dict
                else:
                    logger.warning(f"Unknown object type {out_type}")
                    return

    async def _set_to_standard_cache(self, hashed: str, value: ResponseTypes | dict):
        if self.cache is not None:
            if hasattr(self.cache, "aset"):
                await self.cache.aset(hashed, json.dumps(convert_to_dictionary(value)))
            elif hasattr(self.cache, "set"):
                self.cache.set(hashed, json.dumps(convert_to_dictionary(value)))
            else:
                self.cache[hashed] = json.dumps(convert_to_dictionary(value))

    async def get_from_cache(self, hashed: str, **kwargs) -> Optional[ResponseTypes | dict]:
        if self.cache is not None:
            async with self.semaphore_standard_cache:
                out = await self._get_from_standard_cache(hashed)
            if out is not None:
                # but return in any case
                return out

        # if the item is not in the standard cache, we are trying the semantic cache (if available)
        # we are currently only supporting semantic cache for FAISS models
        if self.semantic_cache is not None:
            logger.debug(f"trying to get {hashed} from semantic cache")
            async with self.semaphore_semantic_cache:
                try:
                    out: Optional[
                        ResponseTypes
                    ] = await self.semantic_cache.query_cache(
                        kwargs,
                        semantic_cache_encoding_method=kwargs.get(
                            "semantic_cache_encoding_method"
                        ),
                    )
                except Exception as e:
                    # we are not stopping on exception here, but logging it instead
                    logger.exception(e)
                    out = None

            if out is not None:
                # propagate to standard cache (we know it is not in standard cache),
                # otherwise we would not end up here
                logger.debug(f"found request {hashed} in semantic cache")

                if self.propagate_semantic_cache_to_standard_cache and out.is_exact:
                    logger.debug("propagating semantic cache to standard cache")
                    await self._set_to_standard_cache(hashed, out)

                return out

    async def set_to_cache(
        self, hashed: str, out: ResponseTypes, request_params: dict[str, Any]
    ) -> None:
        if self.semantic_cache is not None:
            logger.debug(f"adding request {hashed} to semantic cache")
            await self.semantic_cache.add_cache(
                request_params,
                out,
                semantic_cache_encoding_method=request_params.get(
                    "semantic_cache_encoding_method"
                ),
            )

        if self.cache is not None:
            logger.debug(f"adding request {hashed} to standard cache")
            await self._set_to_standard_cache(hashed, out)

    async def retrying_chat_completion(
        self,
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
        **kwargs,
    ) -> Optional[ChatCompletionAddition]:
        logger.debug("requesting chat completion for GPT model")
        try:
            async for attempt in AsyncRetrying(
                retry=(
                    retry_if_exception_type(openai.APIConnectionError)
                    | retry_if_exception_type(openai.RateLimitError)
                    | retry_if_exception_type(openai.InternalServerError)
                    | retry_if_exception_type(asyncio.TimeoutError)
                    | retry_if_exception_type(TimeoutError)
                    # shall also retry for bad gateway error (502)
                    # profile the error when it comes up again
                ),
                wait=wait_random_exponential(min=5, max=60),
                stop=stop_after_attempt(self.stop_after_attempts)
                if self.stop_after_attempts is not None
                else None,
                after=after_logging(
                    logger_level=logging_level, logger_exception=logging_exception
                ),
            ):
                with attempt:
                    out: ChatCompletionAddition = await self.chat_completion(**kwargs)
                    logger.debug("got chat completion for GPT model")
                    return out

        except Exception as e:
            if self.stop_on_exception:
                raise e
            else:
                logger.exception(e)
                return None

    async def cached_chat_completion(
        self,
        content: dict[str, Any],
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
    ) -> Optional[ChatCompletionAddition]:
        # we want to stagger even the cache access a bit, otherwise all requests immediately hit cache
        # but do not see if a later request is putting anything into the cache.
        # Thus, we are limiting the number of parallel executions here

        hashed = make_hash_chatgpt_request(content)

        # accessing cache, returning if available, otherwise proceeding.
        cached_result = await self.get_from_cache(hashed, **content)

        if cached_result is not None:
            return cached_result

        out = await self.retrying_chat_completion(
            logging_level=logging_level, logging_exception=logging_exception, **content
        )

        if out is not None:
            await self.set_to_cache(hashed=hashed, out=out, request_params=content)

        return out

    async def _cached_chat_completion_with_index(
        self, index: int, **kwargs
    ) -> tuple[int, Optional[ChatCompletionAddition] | Exception]:
        try:
            res = await self.cached_chat_completion(**kwargs)
        except Exception as e:
            res = e

        return index, res
