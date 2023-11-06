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
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from lbgpt.cache import make_hash_sha256

logger = getLogger(__name__)


class _BaseGPT(abc.ABC):
    def __init__(self, max_parallel_calls: int, cache: Optional[Any] = None):
        self.cache = cache
        self.max_parallel_calls = max_parallel_calls
        self.semaphore = self.refresh_semaphore()

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

    @retry(
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.TryAgain)
            | retry_if_exception_type(openai.error.APIConnectionError)
        ),
        wait=wait_random_exponential(min=5, max=60),
        stop=stop_after_attempt(10),
        after=after_log(logger, logging.WARNING),
    )
    async def cached_chat_completion(self, **kwargs) -> openai.ChatCompletion:
        if self.cache is not None:
            hashed = make_hash_sha256(kwargs)
            if hashed in self.cache:
                logger.debug("cache hit")
                return self.cache[hashed]

        out = await self.chat_completion(**kwargs)

        if self.cache is not None:
            self.cache[hashed] = out

        return out


class ChatGPT(_BaseGPT):
    def __init__(
        self,
        api_key: str,
        max_parallel_calls: int = 5,
        cache: Optional[Any] = None,
    ):
        super().__init__(cache=cache, max_parallel_calls=max_parallel_calls)
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
    ):
        super().__init__(cache=cache, max_parallel_calls=max_parallel_calls)
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
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls_openai + max_parallel_calls_azure,
        )

        self.openai = ChatGPT(
            api_key=openai_api_key,
            cache=cache,
            max_parallel_calls=max_parallel_calls_openai,
        )

        self.azure = AzureGPT(
            api_key=azure_api_key,
            azure_api_base=azure_api_base,
            azure_model_map=azure_model_map,
            azure_openai_version=azure_openai_version,
            azure_openai_type=azure_openai_type,
            cache=cache,
            max_parallel_calls=max_parallel_calls_azure,
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
            return await self.azure.chat_completion(**kwargs)
