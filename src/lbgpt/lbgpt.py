# -*- coding: utf-8 -*-
import asyncio
import datetime
from asyncio import Timeout
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import openai
from litellm import Router
from litellm.types.router import DeploymentTypedDict
from litellm.types.utils import ModelResponse
from openai._types import NOT_GIVEN, NotGiven

from lbgpt.allocation import (
    max_headroom_allocation_function,
    random_allocation_function,
)
from lbgpt.base import _BaseGPT
from lbgpt.types import ChatCompletionAddition
from lbgpt.usage import Usage

logger = getLogger(__name__)


class ChatGPT(_BaseGPT):
    def __init__(
            self,
            api_key: str,
            max_parallel_calls: int = 5,
            request_timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
            cache: Optional[Any] = None,
            semantic_cache: Optional[Any] = None,
            propagate_semantic_cache_to_standard_cache: bool = False,
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            max_usage_cache_size: Optional[int] = 1_000,
            limit_tpm: Optional[int] = None,
            limit_rpm: Optional[int] = None,
            auto_cache=True,
    ):
        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_usage_cache_size=max_usage_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
            auto_cache=auto_cache,
        )

        self.api_key = api_key
        self.request_timeout = request_timeout

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=None, max_connections=None)
            )
        )

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        # one request to the OpenAI API respecting their ratelimit
        async with (self.semaphore_chatgpt):
            timeout = kwargs.pop("request_timeout", self.request_timeout)

            # removing private parameters that are not being passed to ChatGPT
            kwargs.pop("semantic_cache_encoding_method", None)
            kwargs.pop("model_name_cache_alias", None)

            start = datetime.datetime.now()
            out = (await self.client
                   .with_options(timeout=timeout)
                   .chat.completions.create(
                **kwargs,
            ))

        await self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_chat_completion(out, model_class=self.__class__.__name__)


class AzureGPT(_BaseGPT):
    def __init__(
            self,
            api_key: str,
            azure_api_base: str,
            azure_model_map: dict[str, str],
            cache: Optional[Any] = None,
            semantic_cache: Optional[Any] = None,
            propagate_semantic_cache_to_standard_cache: bool = False,
            azure_openai_version: str = "2024-02-01",
            azure_openai_type: str = "azure",
            max_parallel_calls: int = 5,
            request_timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            max_usage_cache_size: Optional[int] = 1_000,
            limit_tpm: Optional[int] = None,
            limit_rpm: Optional[int] = None,
            auto_cache=True,
    ):
        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_usage_cache_size=max_usage_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
            auto_cache=auto_cache,
        )

        self.api_key = api_key
        self.azure_api_base = azure_api_base
        self.azure_openai_version = azure_openai_version
        self.request_timeout = request_timeout

        self.azure_model_map = azure_model_map

    def get_client(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_api_base,
            api_version=self.azure_openai_version,
            timeout=self.request_timeout,
            max_retries=0,
        )

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        """One request to the Azure OpenAI API respecting their ratelimit
        # needs to change the model parameter to deployment id
        """

        deployment_id = self.azure_model_map[kwargs["model"]]
        kwargs["model"] = deployment_id

        timeout = kwargs.pop("request_timeout", self.request_timeout)

        start = datetime.datetime.now()

        async with self.semaphore_chatgpt:
            out = (
                await self.get_client()
                .with_options(timeout=timeout)
                .chat.completions.create(
                    **kwargs,
                )
            )

        await self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_chat_completion(out, model_class=self.__class__.__name__)


ALLOCATION_FUNCTIONS = {
    "random": random_allocation_function,
    "max_headroom": max_headroom_allocation_function,
}


class MultiLoadBalancedGPT(_BaseGPT):
    def __init__(
            self,
            gpts: list[_BaseGPT],
            allocation_function: str = "random",
            allocation_function_kwargs: Optional[dict] = None,
            allocation_function_weights: Optional[Sequence] = None,
            cache: Optional[Any] = None,
            semantic_cache: Optional[Any] = None,
            propagate_standard_cache_to_semantic_cache: bool = False,
            propagate_semantic_cache_to_standard_cache: bool = False,
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            max_parallel_requests: Optional[int] = None,
            auto_cache=True,
    ):
        self.gpts = gpts

        if isinstance(allocation_function, str):
            allocation_function = ALLOCATION_FUNCTIONS[allocation_function]
        else:
            raise NotImplementedError(
                f"Cannot infer allocation function from type {type(allocation_function)}"
            )

        self.allocation_function = allocation_function
        self.allocation_function_kwargs = allocation_function_kwargs or {}

        if allocation_function_weights is not None:
            assert len(allocation_function_weights) == len(
                gpts
            ), "if provided, `allocation_function_weights` must be the same length as gpts"

        self.allocation_function_weights = allocation_function_weights
        if max_parallel_requests is None:
            max_parallel_requests = sum([gpt.max_parallel_calls for gpt in gpts])

        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_requests,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            auto_cache=auto_cache,
        )

    def request_setup(self):
        for gpt in self.gpts:
            gpt.request_setup()
        super().request_setup()

    @property
    async def usage_cache_list(self) -> list[Usage]:
        out = sum([await gpt.usage_cache_list for gpt in self.gpts], [])
        return out

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        gpt = await self.allocation_function(
            self.gpts,
            weights=self.allocation_function_weights,
            **self.allocation_function_kwargs,
        )

        return await gpt.chat_completion(**kwargs)


class LoadBalancedGPT(MultiLoadBalancedGPT):
    """
    We are continuing to support this for backward compatability reasons, but it is discouraged to use it.
    """

    def __init__(
            self,
            openai_api_key: str,
            azure_api_key: str,
            azure_api_base: str,
            azure_model_map: dict[str, str],
            cache: Optional[Any] = None,
            semantic_cache: Optional[Any] = None,
            propagate_semantic_cache_to_standard_cache: bool = False,
            azure_openai_version: str = "2024-02-01",
            azure_openai_type: str = "azure",
            max_parallel_calls_openai: int = 5,
            max_parallel_calls_azure: int = 5,
            ratio_openai_to_azure: float = 0.25,
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            auto_cache: bool = True,
    ):
        self.openai = ChatGPT(
            api_key=openai_api_key,
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_calls_openai,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            auto_cache=False
        )

        self.azure = AzureGPT(
            api_key=azure_api_key,
            azure_api_base=azure_api_base,
            azure_model_map=azure_model_map,
            azure_openai_version=azure_openai_version,
            azure_openai_type=azure_openai_type,
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_calls_azure,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            auto_cache=False
        )

        super().__init__(
            gpts=[self.openai, self.azure],
            cache=cache,
            semantic_cache=semantic_cache,
            allocation_function="random",
            allocation_function_weights=[
                ratio_openai_to_azure,
                1 - ratio_openai_to_azure,
            ],
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            auto_cache=auto_cache,
        )


class LiteLlmRouter(_BaseGPT):
    def __init__(
            self,
            model_list: list[DeploymentTypedDict | dict[str, Any]],
            max_parallel_calls: int = 5,
            cache: Optional[Any] = None,
            semantic_cache: Optional[Any] = None,
            propagate_semantic_cache_to_standard_cache: bool = False,
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            max_usage_cache_size: Optional[int] = 1_000,
            limit_tpm: Optional[int] = None,
            limit_rpm: Optional[int] = None,
            auto_cache=True,
    ):
        super().__init__(
            cache=cache,
            semantic_cache=semantic_cache,
            propagate_semantic_cache_to_standard_cache=propagate_semantic_cache_to_standard_cache,
            max_parallel_calls=max_parallel_calls,
            stop_after_attempts=stop_after_attempts,
            stop_on_exception=stop_on_exception,
            max_usage_cache_size=max_usage_cache_size,
            limit_tpm=limit_tpm,
            limit_rpm=limit_rpm,
            auto_cache=auto_cache,
        )

        self.router = Router(model_list=model_list, default_max_parallel_requests=max_parallel_calls, num_retries=0)

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        # one request to the OpenAI API respecting their ratelimit
        async with (self.semaphore_chatgpt):
            timeout = kwargs.pop("request_timeout", None)

            # removing private parameters that are not being passed to ChatGPT
            kwargs.pop("semantic_cache_encoding_method", None)
            kwargs.pop("model_name_cache_alias", None)

            start = datetime.datetime.now()
            async with asyncio.timeout(timeout):
                out: ModelResponse = await self.router.acompletion(**kwargs)


        await self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_litellm_model_response(out, model_class=self.__class__.__name__)
