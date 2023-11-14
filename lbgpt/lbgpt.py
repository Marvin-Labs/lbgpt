# -*- coding: utf-8 -*-
import abc
import random
import warnings
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


class ChatGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        max_parallel_calls: int = 5,
        cache: Optional[Any] = None,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )
        self.api_key = api_key

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        # one request to the OpenAI API respecting their ratelimit

        async with self.semaphore:
            return await openai.ChatCompletion.acreate(
                api_key=self.api_key,
                **kwargs,
            )


class AzureGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls: int = 5,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.api_key = api_key
        self.azure_api_base = azure_api_base
        self.azure_model_map = azure_model_map
        self.azure_openai_version = azure_openai_version
        self.azure_openai_type = azure_openai_type

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        """One request to the Azure OpenAI API respecting their ratelimit
        # needs to change the model parameter to deployment id
        """

        model = kwargs.pop("model")
        deployment_id = self.azure_model_map[model]
        kwargs["deployment_id"] = deployment_id

        async with self.semaphore:
            return await openai.ChatCompletion.acreate(
                api_key=self.api_key,
                api_base=self.azure_api_base,
                api_type=self.azure_openai_type,
                api_version=self.azure_openai_version,
                **kwargs,
            )


class LoadBalancedGPT(_BaseGPT):
    def __init__(
        self,
        openai_api_key: str,
        azure_api_key: str,
        azure_api_base: str,
        azure_model_map: dict[str, str],
        cache: Optional[Any] = None,
        azure_openai_version: str = "2023-05-15",
        azure_openai_type: str = "azure",
        max_parallel_calls_openai: int = 5,
        max_parallel_calls_azure: int = 5,
        ratio_openai_to_azure: float = 0.25,
        stop_after_attempts: Optional[int] = 10,
        stop_on_exception: bool = False,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls_openai + max_parallel_calls_azure,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.openai = ChatGPT(
            api_key=openai_api_key,
            cache=cache,
            max_parallel_calls=max_parallel_calls_openai,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.azure = AzureGPT(
            api_key=azure_api_key,
            azure_api_base=azure_api_base,
            azure_model_map=azure_model_map,
            azure_openai_version=azure_openai_version,
            azure_openai_type=azure_openai_type,
            cache=cache,
            max_parallel_calls=max_parallel_calls_azure,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
        )

        self.ratio_openai_to_azure = ratio_openai_to_azure

    async def chat_completion(self, **kwargs) -> openai.ChatCompletion:
        self.openai.refresh_semaphore()
        self.azure.refresh_semaphore()

        if random.random() < self.ratio_openai_to_azure:
            # route to OpenAI API
            return await self.openai.chat_completion(**kwargs)
        else:
            # route to Azure API
            if kwargs["model"] not in self.azure.azure_model_map:
                warnings.warn(
                    "Our Azure API does not support model {kwargs['model']}. Falling back to OpenAI API."
                )
                return await self.openai.chat_completion(**kwargs)
            return await self.azure.chat_completion(**kwargs)
