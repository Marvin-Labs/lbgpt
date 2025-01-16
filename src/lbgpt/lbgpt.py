# -*- coding: utf-8 -*-
import datetime
from asyncio import Timeout
from logging import getLogger
from typing import Any, Optional, Sequence

import httpx
import litellm
import openai
from litellm import Router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.router import DeploymentTypedDict
from litellm.types.utils import ModelResponse
from openai._types import NOT_GIVEN, NotGiven  # noqa
from openai.types import EmbeddingCreateParams

from lbgpt.allocation import (
    max_headroom_allocation_function,
    random_allocation_function,
)
from lbgpt.base import _BaseGPT
from lbgpt.cache import make_hash_embedding_request
from lbgpt.types import ChatCompletionAddition, EmbeddingResponseAddition, EmbeddingAddition
from lbgpt.usage import Usage
from litellm.types.utils import Usage as LitellmUsage


logger = getLogger(__name__)
litellm.suppress_debug_info = True


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
                limits=httpx.Limits(
                    max_keepalive_connections=None, max_connections=None
                )
            ),
        )

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        # one request to the OpenAI API respecting their ratelimit
        async with (self.semaphore_chatgpt):
            timeout = kwargs.pop("request_timeout", self.request_timeout)

            # removing private parameters that are not being passed to ChatGPT
            kwargs.pop("semantic_cache_encoding_method", None)
            kwargs.pop("model_name_cache_alias", None)

            start = datetime.datetime.now()
            out = await self.client.with_options(
                timeout=timeout
            ).chat.completions.create(
                **kwargs,
            )

        await self.add_usage_to_usage_cache(
            Usage(
                input_tokens=out.usage.prompt_tokens,
                output_tokens=out.usage.total_tokens,
                start_datetime=start,
                end_datetime=datetime.datetime.now(),
            )
        )

        return ChatCompletionAddition.from_chat_completion(
            out, model_class=self.__class__.__name__
        )


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

        return ChatCompletionAddition.from_chat_completion(
            out, model_class=self.__class__.__name__
        )


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
            auto_cache=False,
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
            auto_cache=False,
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
            stop_after_attempts: Optional[int] = 10,
            stop_on_exception: bool = False,
            auto_cache=True,
            **kwargs,
    ):
        super().__init__(
            cache=cache,
            max_parallel_calls=max_parallel_calls,
            stop_on_exception=stop_on_exception,
        )

        self.router = Router(
            model_list=model_list,
            default_max_parallel_requests=max_parallel_calls,
            num_retries=stop_after_attempts,
            cache_responses=auto_cache,
            **kwargs,
        )

    def _prepare_private_args(self, request_arguments: dict) -> dict:
        timeout = request_arguments.pop("request_timeout", None)

        # removing private parameters that are not being passed to ChatGPT
        request_arguments.pop("semantic_cache_encoding_method", None)
        request_arguments.pop("model_name_cache_alias", None)

        return {
            **request_arguments,
            "timeout": timeout,
        }

    async def chat_completion(self, **kwargs) -> ChatCompletionAddition:
        # one request to the OpenAI API respecting their ratelimit

        request_arguments = self._prepare_private_args(kwargs)
        out: ModelResponse | CustomStreamWrapper = await self.router.acompletion(
            **request_arguments
        )

        # if the response is a stream, we need to consume it and build it up
        # we are not exposing streamed responses to the user
        if isinstance(out, CustomStreamWrapper):
            out = litellm.stream_chunk_builder(
                [chunk async for chunk in out], messages=kwargs["messages"]
            )

        return ChatCompletionAddition.from_litellm_model_response(
            out, model_class=self.__class__.__name__
        )

    async def embedding(self, model: str, **kwargs) -> EmbeddingResponseAddition:
        request_arguments = self._prepare_private_args(kwargs)

        out = await self.router.aembedding(model=model, **request_arguments)

        return EmbeddingResponseAddition.from_litellm_model_response(
            out, model_class=self.__class__.__name__
        )

    async def cached_embedding(self, model: str, **kwargs) -> EmbeddingResponseAddition:
        # the request accepts a string input, but we want to make sure to convert it to a list
        if isinstance(kwargs['input'], str):
            kwargs['input'] = [kwargs['input']]

        request = EmbeddingCreateParams(model=model, **kwargs)

        inputs = kwargs.pop("input")
        hasheds: list[str] = make_hash_embedding_request(request)

        cached_results = [
            await self._get_from_standard_cache(hashed) for hashed in hasheds
        ]

        logger.info(f'Found {len([k for k in cached_results if k])} of {len(inputs)} in cache')

        missing_results = [input_text for input_text, cached_result in zip(inputs, cached_results) if not cached_result]
        missing_hashed = [hash_ for hash_, cached_result in zip(hasheds, cached_results) if not cached_result]

        model_name = 'unknown'

        usage = LitellmUsage()

        if len(missing_results) > 0:
            logger.debug(f'Fetching {len(missing_results)} missing results')
            missing_request = EmbeddingCreateParams(model=model, input=missing_results, **kwargs)
            missing_response = await self.embedding(**missing_request)

            usage = missing_response.usage

            for hashed, result in zip(missing_hashed, missing_response.data):
                await self._set_to_standard_cache(hashed=hashed, value={
                    'model': model,
                    'embedding': result.embedding,
                    'object': 'embedding'
                })

            model_name = missing_response.model

        elif len(cached_results) > 0:
            model_name = cached_results[0]['model']

        return_data: list[EmbeddingAddition] = list()
        misses = 0
        for i, cached_result in enumerate(cached_results):
            if cached_result:
                # this is a cached result
                return_data.append(EmbeddingAddition(
                    embedding=cached_result['embedding'],
                    is_cached=True,
                    index=i
                ))
            else:
                # this is a cache miss
                return_data.append(
                    EmbeddingAddition(
                        embedding=missing_response.data[misses].embedding, # noqa
                        is_cached=False,
                        index=i
                    )
                )
                misses += 1

        return EmbeddingResponseAddition(
            model_class=self.__class__.__name__,
            data=return_data,
            usage=usage,
            model=model_name,
        )

