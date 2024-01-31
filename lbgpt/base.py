# -*- coding: utf-8 -*-
import abc
import asyncio
import datetime
import json
import logging
import sys
import warnings
from logging import getLogger
from statistics import median
from typing import Any, Callable, Optional, Sequence

import openai
from openai._compat import model_dump, model_parse
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
from lbgpt.semantic_cache.base import _SemanticCacheBase
from lbgpt.types import ChatCompletionAddition
from lbgpt.usage import Usage, UsageStats

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
        propagate_standard_cache_to_semantic_cache: bool = False,
        propagate_semantic_cache_to_standard_cache: bool = False,
    ):
        # this is standard cache, i.e. it only checks for equal items
        self.cache = cache
        self.semantic_cache: _SemanticCacheBase = semantic_cache

        self.max_parallel_calls = max_parallel_calls
        self.stop_after_attempts = stop_after_attempts
        self.stop_on_exception = stop_on_exception

        self._usage_cache_list = []
        self.max_usage_cache_size = max_usage_cache_size

        self.limit_tpm = limit_tpm
        self.limit_rpm = limit_rpm

        self.propagate_standard_cache_to_semantic_cache = (
            propagate_standard_cache_to_semantic_cache
        )
        self.propagate_semantic_cache_to_standard_cache = (
            propagate_semantic_cache_to_standard_cache
        )

        # this is a silly configuration (propagating to semantic cache without having a semantic cache)
        if (
            self.propagate_standard_cache_to_semantic_cache
            and self.semantic_cache is None
        ):
            self.propagate_standard_cache_to_semantic_cache = False
            warnings.warn(
                "propagate_standard_cache_to_semantic_cache is True, but no semantic cache is provided. There will be no propagation."
            )

        # this is a silly configuration (propagating to standard cache without having a standard cache)
        if self.propagate_semantic_cache_to_standard_cache and self.cache is None:
            self.propagate_semantic_cache_to_standard_cache = False
            warnings.warn(
                "propagate_semantic_cache_to_standard_cache is True, but no standard cache is provided. There will be no propagation."
            )



    def request_setup(self):
        pass

    @property
    def usage_cache_list(self) -> list[Usage]:
        return self._usage_cache_list

    def add_usage_to_usage_cache(self, usage: Usage):
        # evict if the list is too long. Do this to protect memory usage if required
        if (
            self.max_usage_cache_size
            and len(self._usage_cache_list) > self.max_usage_cache_size
        ):
            self._usage_cache_list.pop(0)

        self._usage_cache_list.append(usage)

    def usage_cache_list_after_start_datetime(
        self, start_datetime: datetime.datetime
    ) -> list[Usage]:
        return [k for k in self._usage_cache_list if k.start_datetime > start_datetime]

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
            headroom_rpm = self.expected_tokens_per_request() * (
                self.limit_rpm - cur_usage.requests - 1
            )
        else:
            headroom_rpm = sys.maxsize

        return min([headroom_tpm, headroom_rpm])

    @abc.abstractmethod
    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        raise NotImplementedError

    async def chat_completion_list(
        self,
        chatgpt_chat_completion_request_body_list: list[dict],
        show_progress=True,
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
    ) -> Sequence[ChatCompletionAddition]:
        # we are setting up the semaphore here, so that we can refresh it if required

        semaphore = asyncio.Semaphore(self.max_parallel_calls)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(
                self.cached_chat_completion(
                    semaphore=semaphore,
                    logging_level=logging_level,
                    logging_exception=logging_exception,
                    content=content,
                )
            ) for content in chatgpt_chat_completion_request_body_list]

        return [t.result() for t in tasks]

    def _get_from_cache(self, hashed: str) -> Optional[ChatCompletionAddition]:
        if self.cache is not None:
            logger.debug(f'trying to get {hashed} from standard cache')
            out = self.cache.get(hashed)
            if out is not None:
                logger.debug(f'found request {hashed} in standard cache')
                return model_parse(ChatCompletionAddition, json.loads(out))

    def _set_to_cache(self, hashed: str, value: ChatCompletionAddition):
        if self.cache is not None:
            self.cache.set(hashed, json.dumps(model_dump(value)))

    async def get_chat_completion_from_cache(
        self, hashed: str, **kwargs
    ) -> Optional[ChatCompletionAddition]:
        if self.cache is not None:
            out = self._get_from_cache(hashed)
            if out is not None:
                # propagate to semantic cache if required
                if self.propagate_standard_cache_to_semantic_cache:
                    logger.debug("checking if the element exists in semantic cache")
                    try:
                        existing_item = await self.semantic_cache.query_cache(kwargs)
                    except Exception:
                        existing_item = None

                    if existing_item is None:
                        logger.debug("propagating standard cache to semantic cache")
                        await self.semantic_cache.add_cache(kwargs, out)

                # but return in any case
                return out

        # if the item is not in the standard cache, we are trying the semantic cache (if available)
        # we are currently only supporting semantic cache for FAISS models
        if self.semantic_cache is not None:
            logger.debug(f'trying to get {hashed} from semantic cache')
            out: Optional[
                ChatCompletionAddition
            ] = await self.semantic_cache.query_cache(kwargs)
            if out is not None:
                # propagate to standard cache (we know it is not in standard cache),
                # otherwise we would not end up here
                logger.debug(f'found request {hashed} in semantic cache')

                if self.propagate_semantic_cache_to_standard_cache and out.is_exact:
                    logger.debug("propagating semantic cache to standard cache")
                    self._set_to_cache(hashed, out)

                return out

    async def set_chat_completion_to_cache(
        self, hashed: str, out: ChatCompletionAddition, request_params: dict[str, Any]
    ) -> None:
        if self.semantic_cache is not None:
            logger.debug(f"adding request {hashed} to semantic cache")
            await self.semantic_cache.add_cache(request_params, out)

        if self.cache is not None:
            logger.debug(f"adding request {hashed} to standard cache")
            self._set_to_cache(hashed, out)

    async def retrying_chat_completion(
        self,
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
        **kwargs,
    ) -> Optional[ChatCompletionAddition]:

        logger.debug('requesting chat completion for GPT model')
        try:
            async for attempt in AsyncRetrying(
                retry=(
                    retry_if_exception_type(openai.APIConnectionError)
                    | retry_if_exception_type(openai.RateLimitError)
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
                    logger.debug('got chat completion for GPT model')
                    return out

        except Exception as e:
            if self.stop_on_exception:
                raise e
            else:
                logger.exception(e)
                return None

    async def unbound_cached_chat_completion(
        self,
        logging_level: int = logging.WARNING,
        logging_exception: bool = False,
        **kwargs,
    ) -> Optional[ChatCompletionAddition]:
        """same as the bound cached completion but without a semaphore lock"""

        hashed = make_hash_chatgpt_request(kwargs)

        # accessing cache, returning if available, otherwise proceeding.
        cached_result = await self.get_chat_completion_from_cache(hashed, **kwargs)
        if cached_result is not None:
            return cached_result

        out = await self.retrying_chat_completion(
            logging_level=logging_level, logging_exception=logging_exception, **kwargs
        )
        await self.set_chat_completion_to_cache(
            hashed=hashed, out=out, request_params=kwargs
        )

        return out

    async def cached_chat_completion(
        self,
        semaphore: asyncio.Semaphore,
            content: dict[str, Any],
            logging_level: int = logging.WARNING,
        logging_exception: bool = False,

    ) -> Optional[ChatCompletionAddition]:
        # we want to stagger even the cache access a bit, otherwise all requests immediately hit cache
        # but do not see if a later request is putting anything into the cache.
        # Thus, we are limiting the number of parallel executions here

        logger.debug('current semaphore value: ' + str(semaphore._value))

        async with semaphore:
            return await self.unbound_cached_chat_completion(
                logging_level=logging_level,
                logging_exception=logging_exception,
                **content,
            )
